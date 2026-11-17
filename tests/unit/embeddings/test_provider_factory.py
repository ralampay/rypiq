from unittest.mock import Mock

import pytest

from app.providers.config import ProviderSettings
from app.providers.contracts import ChatMessage, ProviderError
from app.providers.factory import create_provider


def settings(provider='local', capability='embedding', **options):
    return ProviderSettings(provider=provider, capability=capability, model='model', embedding_size=2, **options)


def test_local_embedding_only_constructs_embedding_model():
    embedding, inference = Mock(), Mock()
    result = create_provider(settings(), embedding_factory=embedding, inference_factory=inference)
    assert result is embedding.return_value
    embedding.assert_called_once_with(model_path='model', context_size=2048, gpu_layers=0)
    inference.assert_not_called()


def test_local_inference_only_constructs_inference_model():
    embedding, inference = Mock(), Mock()
    value = settings(capability='inference')
    assert create_provider(value, embedding_factory=embedding, inference_factory=inference) is inference.return_value
    inference.assert_called_once_with(value)
    embedding.assert_not_called()


@pytest.mark.parametrize('provider', ['openai', 'llama-server'])
def test_remote_factory_owns_client_and_disables_retries(monkeypatch, provider):
    monkeypatch.setenv('TEST_PROVIDER_KEY', 'test-only-value')
    client_factory, local = Mock(), Mock()
    adapter = create_provider(settings(provider, base_url='http://localhost:1234/v1', api_key_env='TEST_PROVIDER_KEY'),
                              client_factory=client_factory, embedding_factory=local)
    assert client_factory.call_args.kwargs == {
        'base_url': 'http://localhost:1234/v1', 'api_key': 'test-only-value', 'timeout': 60, 'max_retries': 0}
    adapter.close()
    client_factory.return_value.close.assert_called_once()
    local.assert_not_called()


def test_missing_secret_fails_before_client_construction(monkeypatch):
    monkeypatch.delenv('MISSING_PROVIDER_KEY', raising=False)
    client_factory = Mock()
    with pytest.raises(ValueError, match='credential'):
        create_provider(settings('openai', api_key_env='MISSING_PROVIDER_KEY'), client_factory=client_factory)
    client_factory.assert_not_called()


def test_server_factory_cannot_fall_back_to_openai_endpoint():
    client_factory = Mock()
    with pytest.raises(ValueError, match='base URL'):
        create_provider(settings('llama-server'), client_factory=client_factory)
    client_factory.assert_not_called()
