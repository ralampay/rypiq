"""Real SDK over loopback HTTP; no API credentials or paid service required."""

import pytest

from app.models.application import Application
from app.providers.config import settings_for
from app.providers.contracts import ChatMessage, ProviderError
from app.providers.factory import open_capability_provider as open_provider, open_discovery_provider
from app.providers.resolution import discovery_settings_for


@pytest.mark.parametrize('provision_type', ['openai', 'llama-server'])
def test_real_sdk_embeddings_chat_discovery_and_separate_endpoints(server, monkeypatch, provision_type):
    monkeypatch.setenv('TEST_HTTP_MODEL_KEY', 'test-only')
    app = Application(name='test', provision_type=provision_type, embedding_model='remote-model',
                      inference_model='remote-model', embedding_size=2, data={'provider_config': {
                          'embedding': {'base_url': server['url'] + '/embedding/v1', 'api_key_env': 'TEST_HTTP_MODEL_KEY', 'dimensions': 2},
                          'inference': {'base_url': server['url'] + '/inference/v1', 'api_key_env': 'TEST_HTTP_MODEL_KEY'},
                      }})
    with open_provider(settings_for(app, 'embedding')) as provider:
        assert provider.embed(['first', 'second']) == [[1., 0.], [1., 0.]]
    with open_discovery_provider(discovery_settings_for(app, 'embedding')) as discovery:
        assert discovery.list_models()[0].id == 'remote-model'
    with open_provider(settings_for(app, 'inference')) as provider:
        result = provider.chat([ChatMessage('system', 'be brief'), ChatMessage('user', 'hello')], max_tokens=12)
        assert result.text == 'answer'
        assert result.usage['total_tokens'] == 3
    assert [r[0] for r in server['requests']] == [
        '/embedding/v1/embeddings', '/embedding/v1/models', '/inference/v1/chat/completions']
    assert server['requests'][0][1]['dimensions'] == 2
    assert server['requests'][0][2] == 'Bearer test-only'
    token_field = 'max_completion_tokens' if provision_type == 'openai' else 'max_tokens'
    assert server['requests'][-1][1][token_field] == 12


@pytest.mark.parametrize('status', [401, 429, 500])
def test_http_failures_are_safe_and_never_retried(server, status):
    server['status'] = status
    app = Application(name='test', provision_type='llama-server', embedding_model='remote-model',
                      embedding_size=2, data={'provider_config': {'embedding': {'base_url': server['url'] + '/v1'}}})
    with open_provider(settings_for(app, 'embedding')) as provider:
        with pytest.raises(ProviderError) as caught:
            provider.embed(['private text'])
    assert 'secret' not in str(caught.value) and 'private' not in str(caught.value)
    assert len(server['requests']) == 1


def test_timeout_is_safe_and_not_retried(server):
    server['delay'] = 0.1
    app = Application(name='test', provision_type='llama-server', embedding_model='remote-model',
                      data={'provider_config': {'embedding': {'base_url': server['url'] + '/v1', 'timeout': 0.01}}})
    with open_provider(settings_for(app, 'embedding')) as provider:
        with pytest.raises(ProviderError):
            provider.embed(['hello'])
    assert len(server['requests']) == 1
