"""Application-scoped retrieval and prompt construction for chat."""
from unittest.mock import Mock

import pytest

from app import application, cli
from app.exceptions import ValidationError
from app.models.application import Application
from app.models.document import Document
from app.models.embedding import Embedding
from app.models.search import SearchResult
from app.operations.inference.chat import Chat
from app.providers.types.chat import ChatMessage, ChatResult


@pytest.fixture
def resources():
    value = Application(name='docs', embedding_model='embed', inference_model='chat',
                        embedding_provision_type='openai', inference_provision_type='openai', embedding_size=2)
    embedding = Mock(model_name='embed')
    embedding.embed.return_value = [[1., 0.]]
    repository = Mock()
    document = Document(name='Guide', application_id=value.id)
    repository.search_by_vector.return_value = [SearchResult(
        Embedding(document_id=document.id, content='The answer is forty-two.', embedding=[1., 0.]),
        document, 0.1)]
    provider = Mock()
    provider.chat.return_value = ChatResult('forty-two', 'chat', 'stop')
    return dict(application=value, embedding_provider=embedding, repository=repository, provider=provider)


def test_chat_retrieves_context_and_preserves_history(resources):
    messages = [ChatMessage('user', 'Earlier'), ChatMessage('assistant', 'Previous'),
                ChatMessage('user', 'What is the answer?')]
    original = list(messages)
    command = Chat(messages=messages, **resources)
    assert not resources['embedding_provider'].mock_calls
    assert command.execute().text == resources['provider'].chat.return_value.text
    resources['embedding_provider'].embed.assert_called_once_with(['What is the answer?'])
    query = resources['repository'].search_by_vector.call_args.kwargs
    assert query['application_id'] == resources['application'].id and query['limit'] == 5
    sent = resources['provider'].chat.call_args.args[0]
    assert sent[0].role == 'system'
    assert 'insufficient' in sent[0].content.lower()
    assert 'Guide' in sent[-1].content and 'The answer is forty-two.' in sent[-1].content
    assert sent[1:-1] == messages[:-1]
    assert messages == original
    for name in ('provider', 'embedding_provider', 'repository'):
        resources[name].close.assert_not_called()


def test_empty_context_and_custom_builder(resources):
    resources['repository'].search_by_vector.return_value = []
    messages = [ChatMessage('user', 'Question')]
    assert Chat(messages=messages, **resources).execute().sources == []
    assert 'No relevant passages' in resources['provider'].chat.call_args.args[0][-1].content
    builder = Mock(return_value=[ChatMessage('user', 'Custom context')])
    Chat(messages=messages, prompt_builder=builder, **resources).execute()
    builder.assert_called_once_with(messages, [])
    assert resources['provider'].chat.call_args.args[0] == builder.return_value


@pytest.mark.parametrize('messages', [[], [ChatMessage('assistant', 'No question')],
                                      [ChatMessage('user', ' ')]])
def test_invalid_messages_fail_before_retrieval(resources, messages):
    with pytest.raises(ValidationError):
        Chat(messages=messages, **resources).execute()
    resources['embedding_provider'].embed.assert_not_called()
    resources['provider'].chat.assert_not_called()


def test_retrieval_failure_prevents_inference(resources):
    resources['repository'].search_by_vector.side_effect = RuntimeError('search failed')
    with pytest.raises(RuntimeError, match='search failed'):
        Chat(messages=[ChatMessage('user', 'Question')], **resources).execute()
    resources['provider'].chat.assert_not_called()
    resources['embedding_provider'].close.assert_not_called()


@pytest.mark.parametrize('args', [[], ['--model', 'model'], ['--system', 'brief'],
                                 ['--max-tokens', '12'], ['--temperature', '0.2']])
def test_chat_cli_requires_application_and_rejects_removed_flags(monkeypatch, args):
    execute = Mock()
    monkeypatch.setattr(application, 'chat', execute)
    from uuid import uuid4
    selection = ['--application-id', str(uuid4())] if args else []
    with pytest.raises(SystemExit) as error:
        cli.main(['chat', 'question', *selection, *args])
    assert error.value.code == 2
    execute.assert_not_called()


def test_prompt_preserves_rank_and_identifies_chunks(resources):
    from app.operations.inference.prompts import build_rag_messages
    first = resources['repository'].search_by_vector.return_value[0]
    second = SearchResult(Embedding(document_id=first.document.id, content='Second passage',
                                    embedding=[0., 1.], chunk_index=1), first.document, 0.2)
    result = build_rag_messages([ChatMessage('user', 'Question')], [first, second])
    text = result[-1].content
    assert text.index(first.embedding.content) < text.index(second.embedding.content)
    assert str(first.embedding.id) in text and str(second.embedding.id) in text


def test_embedding_mismatch_prevents_inference(resources):
    from app.exceptions import ModelMismatchError
    resources['embedding_provider'].model_name = 'wrong'
    with pytest.raises(ModelMismatchError):
        Chat(messages=[ChatMessage('user', 'Question')], **resources).execute()
    resources['embedding_provider'].embed.assert_not_called()
    resources['provider'].chat.assert_not_called()


@pytest.mark.parametrize('failure', [None, 'embedding', 'inference', 'retrieval', 'generation'])
def test_configured_chat_releases_owned_resources(monkeypatch, resources, failure):
    from contextlib import contextmanager
    from unittest.mock import MagicMock
    database = MagicMock()
    database.__enter__.return_value = database
    database.applications.return_value.get.return_value = resources['application']
    database.embeddings.return_value = resources['repository']
    monkeypatch.setattr(application, 'get_database', lambda config=None: database)
    opened, closed = [], []

    @contextmanager
    def open_provider(settings, *, credentials=None):
        capability = settings.capability
        if failure == capability:
            raise RuntimeError('failed')
        opened.append(capability)
        try:
            yield resources['embedding_provider' if capability == 'embedding' else 'provider']
        finally:
            closed.append(capability)

    monkeypatch.setattr(application, 'open_provider', open_provider)
    if failure == 'retrieval':
        resources['repository'].search_by_vector.side_effect = RuntimeError('failed')
    if failure == 'generation':
        resources['provider'].chat.side_effect = RuntimeError('failed')
    kwargs = dict(messages=[ChatMessage('user', 'Question')], application_id=resources['application'].id)
    if failure:
        with pytest.raises(RuntimeError, match='failed'):
            application.chat(**kwargs)
    else:
        result = application.chat(**kwargs)
        assert result.text == resources['provider'].chat.return_value.text
        assert result.sources[0].document_id == resources['repository'].search_by_vector.return_value[0].document.id
    assert closed == opened[::-1]
    database.__exit__.assert_called_once()


def test_configured_invalid_request_does_not_open_database(monkeypatch):
    database = Mock()
    monkeypatch.setattr(application, 'get_database', database)
    from uuid import uuid4
    with pytest.raises(ValidationError):
        application.chat(messages=[ChatMessage('assistant', 'No question')], application_id=uuid4())
    database.assert_not_called()


def test_missing_application_prevents_provider_creation(monkeypatch):
    from unittest.mock import MagicMock
    from uuid import uuid4
    from app.exceptions import ApplicationNotFoundError
    database = MagicMock()
    database.__enter__.return_value = database
    database.applications.return_value.get.return_value = None
    provider = Mock()
    monkeypatch.setattr(application, 'get_database', lambda config=None: database)
    monkeypatch.setattr(application, 'open_provider', provider)
    with pytest.raises(ApplicationNotFoundError):
        application.chat(messages=[ChatMessage('user', 'Question')], application_id=uuid4())
    provider.assert_not_called()
    database.__exit__.assert_called_once()


def test_chat_sources_are_unique_documents_in_retrieval_order(resources):
    from dataclasses import asdict
    first = resources['repository'].search_by_vector.return_value[0]
    first.document.filename = 'guide.txt'
    first.document.source = 'https://example.com/guide'
    second_document = Document(name=first.document.name)
    second = SearchResult(Embedding(document_id=second_document.id, content='Other passage',
                                    embedding=[1., 0.]), second_document, 0.2)
    duplicate = SearchResult(Embedding(document_id=first.document.id, content='Another chunk',
                                       embedding=[1., 0.]), first.document, 0.3)
    resources['repository'].search_by_vector.return_value = [first, second, duplicate]
    answer = ChatResult('Insufficient evidence.', 'chat', 'stop', {'total_tokens': 17})
    resources['provider'].chat.return_value = answer

    result = Chat(messages=[ChatMessage('user', 'Question')], **resources).execute()

    assert isinstance(result, ChatResult)
    assert {key: value for key, value in asdict(result).items() if key != 'sources'} == asdict(answer)
    assert [asdict(source) for source in result.sources] == [
        dict(document_id=first.document.id, name='Guide', filename='guide.txt',
             source='https://example.com/guide', indicator='[1]'),
        dict(document_id=second_document.id, name='Guide', filename='', source='', indicator='[2]'),
    ]
    resources['repository'].search_by_vector.assert_called_once()
    assert not hasattr(answer, 'sources')


@pytest.mark.parametrize('empty', [False, True])
def test_chat_cli_serializes_sources(monkeypatch, capsys, resources, empty):
    import json
    if empty:
        resources['repository'].search_by_vector.return_value = []
    result = Chat(messages=[ChatMessage('user', 'Question')], **resources).execute()
    monkeypatch.setattr(application, 'chat', lambda **kwargs: result)

    cli.main(['chat', 'Question', '--application-id', str(resources['application'].id)])

    payload = json.loads(capsys.readouterr().out)
    assert payload == dict(text='forty-two', model='chat', finish_reason='stop', usage=None,
                          sources=[] if empty else [
                              dict(document_id=str(resources['repository'].search_by_vector.return_value[0].document.id),
                                   name='Guide', filename='', source='', indicator='[1]')])


def test_prompt_citations_match_structured_sources(resources):
    import json
    first = resources['repository'].search_by_vector.return_value[0]
    other = Document(name=first.document.name)
    second = SearchResult(Embedding(document_id=other.id, content='Other text',
                                    embedding=[1., 0.]), other, 0.2)
    resources['repository'].search_by_vector.return_value = [first, second, first]
    markdown = '**Answer:** “The answer is forty-two.” [1]\n\nOther text. [2]'
    resources['provider'].chat.return_value = ChatResult(markdown, 'chat', 'stop')

    result = Chat(messages=[ChatMessage('user', 'Question')], **resources).execute()

    sent = resources['provider'].chat.call_args.args[0]
    passages = json.loads(sent[-1].content.split('Reference passages (JSON):\n', 1)[1].split('\n\nQuestion:', 1)[0])
    assert [passage['indicator'] for passage in passages] == ['[1]', '[2]', '[1]']
    assert [passage['document_id'] for passage in passages] == [str(first.document.id), str(other.id), str(first.document.id)]
    assert [source.indicator for source in result.sources] == ['[1]', '[2]']
    assert result.text == markdown
    assert 'Markdown' in sent[0].content
    assert 'quote' in sent[0].content.lower()
    assert 'cite' in sent[0].content.lower()


def test_empty_retrieval_prompt_does_not_invent_citations(resources):
    resources['repository'].search_by_vector.return_value = []
    result = Chat(messages=[ChatMessage('user', 'Question')], **resources).execute()
    system = resources['provider'].chat.call_args.args[0][0].content
    assert 'Do not invent' in system
    assert 'no reference passages' in system
    assert result.sources == []
