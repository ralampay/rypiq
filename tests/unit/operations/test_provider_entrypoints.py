from contextlib import contextmanager
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock
from uuid import uuid4
import json

from app import application, cli
from app.models.application import Application
from app.providers.contracts import ChatResult


def test_chat_cli_dispatches_to_application(monkeypatch, capsys):
    fake = Mock(return_value=ChatResult('answer', 'model', 'stop'))
    monkeypatch.setattr(application, 'chat', fake, raising=False)
    id = uuid4()
    cli.main(['chat', 'hello', '--application-id', str(id), '--system', 'Be brief'])
    assert json.loads(capsys.readouterr().out)['text'] == 'answer'
    assert fake.call_args.kwargs['application_id'] == id


def test_model_list_cli_supports_catalog_without_database(monkeypatch, capsys):
    fake = Mock(return_value=[])
    monkeypatch.setattr(application, 'list_models', fake, raising=False)
    cli.main(['models:list'])
    assert json.loads(capsys.readouterr().out) == []
    assert fake.call_args.kwargs['discover'] is False


def test_remote_search_uses_application_endpoint_and_no_local_model(monkeypatch):
    app = Application(name='a', embedding_model='remote-model', provision_type='llama-server', embedding_size=2,
                      data={'provider_config': {'embedding': {'base_url': 'http://localhost:8001/v1'}}})
    database = MagicMock()
    database.__enter__.return_value = database
    database.applications.return_value.get.return_value = app
    database.embeddings.return_value.search_by_vector.return_value = []
    monkeypatch.setattr(application, 'get_database', lambda: database)
    model = Mock(model_name='remote-model')
    model.embed.return_value = [[1., 0.]]
    captured = []
    @contextmanager
    def open_provider(settings, **kwargs):
        captured.append(settings)
        yield model
    monkeypatch.setattr(application, 'open_provider', open_provider, raising=False)
    assert application.search_documents(query='hello', application_id=app.id) == []
    assert captured[0].base_url == 'http://localhost:8001/v1'
    model.embed.assert_called_once_with(['hello'])
    database.embeddings.return_value.search_by_vector.assert_called_once_with(
        embedding=[1., 0.], application_id=app.id, limit=10, metadata_filters=None)


def test_standalone_embedding_uses_catalog_runtime_defaults(tmp_path, monkeypatch):
    import yaml
    source = tmp_path / 'input.txt'
    source.write_text('hello')
    catalog = tmp_path / 'models.yml'
    catalog.write_text(yaml.safe_dump({'models': {'embed': {'provider': 'local',
        'capabilities': ['embedding'], 'path': 'weights.gguf',
        'options': {'context_size': 4096, 'gpu_layers': 2, 'document_prefix': ''}}}}))
    monkeypatch.setenv('RYPIQ_MODELS_CONFIG', str(catalog))
    provider = Mock(model_name='weights.gguf')
    provider.embed.return_value = [[1., 0.]]
    factory = Mock(return_value=provider)
    application.embed_file(path=str(source), chunk_size=10, stride=10, model_path='@embed', provider_factory=factory)
    assert factory.call_args.kwargs['context_size'] == 4096
    assert factory.call_args.kwargs['gpu_layers'] == 2
    provider.embed.assert_called_once_with(['hello'])


import pytest
from dataclasses import replace
from app.providers.contracts import ChatMessage, ModelInfo


@pytest.fixture
def configured_backend(monkeypatch):
    database = MagicMock()
    database.__enter__.return_value = database
    monkeypatch.setattr(application, 'get_database', lambda: database)
    return database


@pytest.mark.parametrize('options', [{'model_path': 'wrong.gguf'}, {'context_size': 0}, {'gpu_layers': -2}])
def test_local_application_overrides_are_validated(options):
    with pytest.raises(ValueError):
        application._application_settings(Application(name='a', embedding_model='correct.gguf'), 'embedding', **options)


def test_valid_local_overrides_and_remote_override_rejection():
    value = Application(name='a', embedding_model='model.gguf')
    result = application._application_settings(value, 'embedding', model_path='another/model.gguf',
                                               context_size=512, gpu_layers=-1)
    assert (result.model, result.context_size, result.gpu_layers) == ('another/model.gguf', 512, -1)
    remote = replace(value, provision_type='openai')
    with pytest.raises(ValueError):
        application._application_settings(remote, 'embedding', model_path='model.gguf')
    with pytest.raises(ValueError):
        application._application_settings(remote, 'embedding', context_size=512)


def test_chat_application_and_standalone_borrow_provider(tmp_path, monkeypatch, configured_backend):
    value = Application(name='a', embedding_model='embed', inference_model='chat')
    configured_backend.applications.return_value.get.return_value = value
    model = Mock()
    expected = ChatResult('answer', 'chat', 'stop')
    model.chat.return_value = expected
    settings = []
    @contextmanager
    def open_provider(value, **kwargs):
        settings.append(value)
        yield model
    monkeypatch.setattr(application, 'open_provider', open_provider)
    messages = [ChatMessage('assistant', 'previous'), ChatMessage('user', 'next')]
    assert application.chat(messages=messages, application_id=value.id) is expected
    assert settings[-1].model == 'chat'
    assert application.chat(messages=messages, model_path='direct.gguf', temperature=0.2) is expected
    assert settings[-1].model == 'direct.gguf'
    model.close.assert_not_called()
    configured_backend.__exit__.assert_called()


@pytest.mark.parametrize('options', [{}, {'application_id': uuid4(), 'model_path': 'model'}])
def test_chat_requires_exactly_one_selection(options):
    with pytest.raises(ValueError):
        application.chat(messages=[ChatMessage('user', 'hello')], **options)


def test_chat_missing_inference_fails_before_provider(monkeypatch, configured_backend):
    value = Application(name='a', embedding_model='embed')
    configured_backend.applications.return_value.get.return_value = value
    provider = Mock()
    monkeypatch.setattr(application, 'open_provider', provider)
    with pytest.raises(ValueError):
        application.chat(messages=[ChatMessage('user', 'hello')], application_id=value.id)
    provider.assert_not_called()
    configured_backend.__exit__.assert_called_once()


def test_model_listing_modes_and_endpoint_deduplication(monkeypatch, configured_backend):
    assert application.list_models()
    configured_backend.__enter__.assert_not_called()
    with pytest.raises(ValueError):
        application.list_models(discover=True)
    local = Application(name='a', embedding_model='local')
    configured_backend.applications.return_value.get.return_value = local
    assert all(model.provider == 'local' for model in application.list_models(application_id=local.id, discover=True))
    remote = replace(local, provision_type='openai')
    configured_backend.applications.return_value.get.return_value = remote
    settings = []
    @contextmanager
    def open_provider(value, **kwargs):
        settings.append(value)
        model = Mock()
        model.list_models.return_value = [ModelInfo('remote', 'openai', (), 'discovered', 'available', 'remote')]
        yield model
    monkeypatch.setattr(application, 'open_provider', open_provider)
    assert application.list_models(application_id=remote.id, discover=False)
    assert not settings
    application.list_models(application_id=remote.id, discover=True)
    assert len(settings) == 1  # No inference configured.
    settings.clear()
    remote.inference_model = 'chat'
    application.list_models(application_id=remote.id, discover=True)
    assert len(settings) == 1  # Shared endpoint is queried once.
    settings.clear()
    remote.inference_model = None
    remote.data = {'provider_config': {'inference': {'base_url': 'http://localhost:8123/v1'}}}
    application.list_models(application_id=remote.id, discover=True)
    assert len(settings) == 2  # Inference endpoint can be discovered before selecting its model.


def test_create_and_update_snapshot_catalog_aliases(tmp_path, monkeypatch, configured_backend):
    import yaml
    path = tmp_path / 'models.yml'
    path.write_text(yaml.safe_dump({'models': {'alias': {'provider': 'local',
        'capabilities': ['embedding'], 'path': 'weights.gguf', 'options': {'query_prefix': ''}}}}))
    monkeypatch.setenv('RYPIQ_MODELS_CONFIG', str(path))
    configured_backend.settings = {'dimensions': 2}
    repository = configured_backend.applications.return_value
    repository.create.side_effect = lambda value: value
    repository.update.side_effect = lambda value: value
    created = application.create_application(name='test', embedding_model='@alias')
    assert created.embedding_model == str(tmp_path / 'weights.gguf')
    assert created.embedding_size == 2
    updated = application.update_application(application=replace(created, embedding_model='@alias'))
    assert updated.data['provider_config']['embedding']['query_prefix'] == ''


def test_application_update_cli_merges_fields_and_redacts_key(monkeypatch, capsys):
    value = Application(name='a', embedding_model='embed', inference_model='chat')
    monkeypatch.setattr(application, 'show_application', lambda **kwargs: value)
    update = Mock(side_effect=lambda **kwargs: kwargs['application'])
    monkeypatch.setattr(application, 'update_application', update)
    cli.main(['applications:update', str(value.id), '--name', 'changed', '--clear-inference-model', '--data', '{}'])
    result = json.loads(capsys.readouterr().out)
    assert result['name'] == 'changed' and result['inference_model'] is None
    assert 'api_key' not in result
    cli.main(['applications:update', str(value.id), '--inference-model', '@new-chat'])
    assert json.loads(capsys.readouterr().out)['inference_model'] == '@new-chat'


def test_chat_cli_without_system_and_model_list_serialization(monkeypatch, capsys):
    monkeypatch.setattr(application, 'chat', lambda **kwargs: ChatResult('text', 'model', 'stop'))
    cli.main(['chat', 'hi', '--model', 'model.gguf'])
    assert json.loads(capsys.readouterr().out)['text'] == 'text'
    monkeypatch.setattr(application, 'list_models', lambda **kwargs: [ModelInfo('id', 'openai', (), 'discovered', 'available', 'id')])
    cli.main(['models:list'])
    assert json.loads(capsys.readouterr().out)[0]['capabilities'] == []


def test_embedding_compatibility_function_honors_catalog_defaults(tmp_path, monkeypatch):
    import yaml
    from app.application import embed_text_file
    source = tmp_path / 'input.txt'
    source.write_text('hello')
    path = tmp_path / 'models.yml'
    path.write_text(yaml.safe_dump({'models': {'embed': {'provider': 'local',
        'capabilities': ['embedding'], 'path': 'weights.gguf', 'options': {'context_size': 4096}}}}))
    monkeypatch.setenv('RYPIQ_MODELS_CONFIG', str(path))
    provider = Mock(model_name='weights.gguf')
    provider.embed.return_value = [[1., 0.]]
    factory = Mock(return_value=provider)
    embed_text_file(path=str(source), chunk_size=10, stride=10, model_path='@embed', provider_factory=factory)
    assert factory.call_args.kwargs['context_size'] == 4096


def test_legacy_ingestion_honors_saved_runtime_options(tmp_path, monkeypatch):
    source = tmp_path / 'input.txt'
    source.write_text('hello')
    value = Application(name='a', embedding_model='weights.gguf', embedding_size=2,
                        data={'provider_config': {'embedding': {'context_size': 4096, 'gpu_layers': 2}}})
    fake = Mock(return_value={'model': 'weights.gguf', 'chunks': [{'text': 'hello', 'embedding': [1., 0.], 'index': 0, 'start': 0, 'end': 5}]})
    monkeypatch.setattr(application, 'embed_text_file', fake)
    application.ingest_document(str(source), Mock(), Mock(name_exists=Mock(return_value=False)), 10, 10, application=value, name='test document')
    assert fake.call_args.kwargs['context_size'] == 4096
    assert fake.call_args.kwargs['gpu_layers'] == 2


def test_legacy_local_composition_rejects_remote_before_effects():
    from app.application import ingest_document
    storage, repository = Mock(name_exists=Mock(return_value=False)), Mock()
    value = Application(name='a', embedding_model='remote', provision_type='openai')
    with pytest.raises(ValueError, match='injected provider'):
        ingest_document('unused.txt', storage, repository, 10, 10, application=value, name='test document')
    assert not storage.mock_calls and not repository.mock_calls


@pytest.mark.parametrize('provider_type', ['local', 'openai', 'llama-server'])
def test_mismatched_provider_identity_never_persists_or_searches(tmp_path, provider_type):
    from app.operations.documents.ingest import IngestDocument
    from app.operations.documents.search import SearchDocuments
    source = tmp_path / 'file.txt'
    source.write_text('hello')
    value = Application(name='a', embedding_model='configured', provision_type=provider_type, embedding_size=2)
    model = Mock(model_name='wrong')
    model.embed.return_value = [[1., 0.]]
    storage, repository = Mock(name_exists=Mock(return_value=False)), Mock()
    with pytest.raises(ValueError, match='model must match'):
        IngestDocument(path=str(source), chunk_size=10, stride=10, application=value,
                       embedding_provider=model, storage=storage, repository=repository, name='test document').execute()
    storage.save.assert_not_called()
    repository.create.assert_not_called()
    model.embed.assert_not_called()
    model.close.assert_not_called()
    with pytest.raises(ValueError, match='model must match'):
        SearchDocuments(query='hello', application=value, embedding_provider=model, repository=repository).execute()
    model.embed.assert_not_called()
    repository.search_by_vector.assert_not_called()


def test_direct_models_do_not_require_catalog_file(tmp_path, monkeypatch, configured_backend):
    catalog = Mock(side_effect=AssertionError('direct models must not require the optional catalog'))
    monkeypatch.setattr(application, 'load_catalog', catalog)
    source = tmp_path / 'input.txt'
    source.write_text('hello')
    model = Mock(model_name='direct.gguf')
    model.embed.return_value = [[1., 0.]]
    application.embed_file(path=str(source), chunk_size=10, stride=10, model_path='direct.gguf',
                           provider_factory=lambda **kwargs: model)
    configured_backend.settings = {'dimensions': 2}
    configured_backend.applications.return_value.create.side_effect = lambda value: value
    configured_backend.applications.return_value.update.side_effect = lambda value: value
    created = application.create_application(name='direct', embedding_model='direct.gguf', inference_model='chat.gguf')
    application.update_application(application=created)
    application._application_settings(created, 'embedding', model_path='other/direct.gguf')
    @contextmanager
    def open_provider(*args, **kwargs):
        yield Mock(chat=Mock(return_value=ChatResult('text', 'chat', 'stop')))
    monkeypatch.setattr(application, 'open_provider', open_provider)
    assert application.chat(messages=[ChatMessage('user', 'hello')], model_path='chat.gguf').text == 'text'
    catalog.assert_not_called()


@pytest.mark.parametrize('provider', ['local', 'openai'])
def test_create_snapshots_direct_local_paths(tmp_path, monkeypatch, configured_backend, provider):
    monkeypatch.chdir(tmp_path)
    repository = configured_backend.applications.return_value
    repository.create.side_effect = lambda value: value
    result = application.create_application(name='a', embedding_model='models/embed.gguf',
        inference_model='models/chat.gguf', embedding_size=768, provision_type=provider)
    monkeypatch.chdir(tmp_path.parent)
    prefix = str(tmp_path) + '/' if provider == 'local' else ''
    assert result.embedding_model == prefix + 'models/embed.gguf'
    assert result.inference_model == prefix + 'models/chat.gguf'


def test_shared_session_across_embedding_ingestion_search_and_chat(tmp_path, monkeypatch, configured_backend):
    from app.providers.session import LocalModelSession
    from app.application import embed_text_file, ingest_document
    from app.providers.config import settings_for
    from app.storage.providers.local import LocalStorage

    value = Application(name='a', embedding_model='weights.gguf', inference_model='chat.gguf', embedding_size=2)
    configured_backend.applications.return_value.get.return_value = value
    configured_backend.embeddings.return_value.search_by_vector.return_value = []
    configured_backend.documents.return_value.name_exists.return_value = False
    source = tmp_path / 'input.txt'
    source.write_text('hello')
    storage = LocalStorage(root=tmp_path / 'objects')
    monkeypatch.setattr('app.storage.factory.get_storage', lambda: storage)
    embedding = Mock(model_name='weights.gguf')
    embedding.embed.return_value = [[1., 0.]]
    inference = Mock()
    inference.chat.return_value = ChatResult('answer', 'chat.gguf', 'stop')
    embedding_factory, inference_factory = Mock(return_value=embedding), Mock(return_value=inference)
    with LocalModelSession(embedding_factory=embedding_factory, inference_factory=inference_factory) as session:
        session.warmup(settings_for(value, 'embedding'), settings_for(value, 'inference'))
        for _ in range(2):
            embed_text_file(str(source), 10, 10, model_path=value.embedding_model, model_session=session)
            application.ingest_file(path=str(source), chunk_size=10, stride=10,
                                    application_id=value.id, model_session=session, name='test document')
            ingest_document(str(source), storage, Mock(name_exists=Mock(return_value=False)), 10, 10, application=value, model_session=session, name='test document')
            assert application.search_documents(query='hello', application_id=value.id, model_session=session) == []
            assert application.chat(messages=[ChatMessage('user', 'hi')], application_id=value.id,
                                    model_session=session).text == 'answer'
        embedding_factory.assert_called_once()
        inference_factory.assert_called_once()
        embedding.close.assert_not_called()
        inference.close.assert_not_called()
    embedding.close.assert_called_once()
    inference.close.assert_called_once()


def test_session_rejects_per_call_embedding_factory_before_loading(tmp_path):
    from app.providers.session import LocalModelSession
    source = tmp_path / 'input.txt'
    source.write_text('hello')
    factory = Mock()
    with LocalModelSession(embedding_factory=factory) as session:
        with pytest.raises(ValueError, match='factory'):
            application.embed_file(path=str(source), chunk_size=10, stride=10,
                                   provider_factory=factory, model_session=session)
    factory.assert_not_called()


def test_remote_chat_bypasses_local_session(monkeypatch, configured_backend):
    value = Application(name='a', embedding_model='remote', inference_model='chat', provision_type='openai')
    configured_backend.applications.return_value.get.return_value = value
    provider = Mock()
    provider.chat.return_value = ChatResult('answer', 'chat', 'stop')
    @contextmanager
    def open_remote(settings):
        try:
            yield provider
        finally:
            provider.close()
    monkeypatch.setattr(application, 'open_provider', open_remote)
    session = Mock()
    application.chat(messages=[ChatMessage('user', 'hi')], application_id=value.id, model_session=session)
    session.open_provider.assert_not_called()
    provider.close.assert_called_once()
