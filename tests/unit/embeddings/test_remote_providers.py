from unittest.mock import Mock

import pytest

from app.providers.config import ProviderSettings
from app.providers.contracts import ChatMessage, ProviderError
from app.providers.remote import OpenAIProvider


def provider(capability='embedding', **options):
    client = Mock()
    settings = ProviderSettings(provider='openai', capability=capability, model='model', embedding_size=2, **options)
    return OpenAIProvider(settings, client=client), client


def test_embeddings_reorder_validate_and_pass_explicit_dimensions():
    adapter, client = provider(dimensions=2)
    client.embeddings.create.return_value = {'data': [
        {'index': 1, 'embedding': [0., 1.]}, {'index': 0, 'embedding': [1., 0.]}]}
    assert adapter.embed(['a', 'b']) == [[1., 0.], [0., 1.]]
    client.embeddings.create.assert_called_once_with(model='model', input=['a', 'b'], dimensions=2)
    assert adapter.embed([]) == []


@pytest.mark.parametrize('data', [[], [{'index': 1, 'embedding': [1., 0.]}],
    [{'index': 0, 'embedding': [0., 0.]}], [{'index': 0, 'embedding': [1.]}],
    [{'index': 0, 'embedding': [float('nan'), 0.]}], None])
def test_malformed_embedding_response_is_safe(data):
    adapter, client = provider()
    client.embeddings.create.return_value = {'data': data}
    with pytest.raises(ProviderError, match='response'):
        adapter.embed(['text'])


def test_chat_normalizes_usage_and_omits_unspecified_temperature():
    adapter, client = provider('inference')
    client.chat.completions.create.return_value = {
        'model': 'reported', 'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 2, 'completion_tokens': 3, 'total_tokens': 5}}
    result = adapter.chat([ChatMessage('user', 'hello')], max_tokens=10)
    assert result.text == 'answer'
    assert result.model == 'reported'
    assert result.usage['total_tokens'] == 5
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs['max_completion_tokens'] == 10
    assert 'temperature' not in kwargs
    assert kwargs['messages'] == [{'role': 'user', 'content': 'hello'}]


@pytest.mark.parametrize('response', [{}, {'choices': []}, {'choices': [{'message': {'content': None}}]},
    {'choices': [{'message': {'content': 'text'}}], 'usage': {'total_tokens': -1}}])
def test_malformed_chat_response_is_safe(response):
    adapter, client = provider('inference')
    client.chat.completions.create.return_value = response
    with pytest.raises(ProviderError):
        adapter.chat([ChatMessage('user', 'hello')])


def test_transport_failure_does_not_expose_credentials():
    adapter, client = provider()
    client.embeddings.create.side_effect = RuntimeError('secret upstream body')
    with pytest.raises(ProviderError) as caught:
        adapter.embed(['private prompt'])
    assert 'secret' not in str(caught.value)
    assert 'private' not in str(caught.value)


def test_discovery_does_not_guess_capabilities():
    adapter, client = provider()
    client.models.list.return_value = {'data': [{'id': 'z'}, {'id': 'a'}, {'id': 'z'}]}
    models = adapter.list_models()
    assert [model.id for model in models] == ['a', 'z']
    assert all(model.capabilities == () for model in models)
