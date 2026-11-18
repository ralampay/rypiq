"""Public capability boundaries and compatibility imports."""
from unittest.mock import Mock

from app.providers.config import ProviderSettings
from app.providers.factory import create_provider, create_embedding_provider, create_inference_provider


def test_embedding_factory_returns_only_embedding_capability():
    settings = ProviderSettings('llama-server', 'embedding', 'model', 2,
                                base_url='http://localhost:8080/v1')
    provider = create_embedding_provider(settings, client_factory=Mock())
    assert callable(provider.embed)
    assert not hasattr(provider, 'chat')
    assert not hasattr(provider, 'list_models')


def test_inference_factory_returns_only_inference_capability():
    settings = ProviderSettings('llama-server', 'inference', 'model', 2,
                                base_url='http://localhost:8080/v1')
    provider = create_inference_provider(settings, client_factory=Mock())
    assert callable(provider.chat)
    assert not hasattr(provider, 'embed')
    assert not hasattr(provider, 'list_models')


def test_contract_and_value_compatibility_imports_preserve_identity():
    from app.embeddings.base import EmbeddingProvider as LegacyEmbedding
    from app.embeddings.local import LocalEmbeddingProvider as LegacyLocal
    from app.providers import contracts
    from app.providers.contracts.embedding import EmbeddingProvider
    from app.providers.contracts.inference import InferenceProvider
    from app.providers.contracts.discovery import ModelDiscovery
    from app.providers.local.embedding import LocalEmbeddingProvider
    from app.providers.types.chat import ChatMessage, ChatResult
    from app.providers.types.model import ModelInfo

    assert LegacyEmbedding is contracts.EmbeddingProvider is EmbeddingProvider
    assert LegacyLocal is LocalEmbeddingProvider
    assert contracts.InferenceProvider is InferenceProvider
    assert contracts.ModelDiscovery is ModelDiscovery
    assert contracts.ChatMessage is ChatMessage
    assert contracts.ChatResult is ChatResult
    assert contracts.ModelInfo is ModelInfo


def test_contract_imports_do_not_load_infrastructure():
    import subprocess
    import sys

    subprocess.run([sys.executable, '-c', '''
import sys
import app.providers.contracts
for name in ('openai', 'llama_cpp', 'sqlalchemy', 'yaml',
             'app.providers.factory', 'app.providers.config', 'app.models.application'):
    assert name not in sys.modules, name
'''], check=True)


def test_discovery_constructs_without_model_and_closes_owned_client():
    from app.providers.config import DiscoverySettings
    from app.providers.factory import open_discovery_provider

    client = Mock()
    client.models.list.return_value = {'data': [{'id': 'model'}]}
    settings = DiscoverySettings('llama-server', base_url='http://localhost:8080/v1')
    with open_discovery_provider(settings, client_factory=Mock(return_value=client)) as provider:
        assert not hasattr(provider, 'embed')
        assert not hasattr(provider, 'chat')
        assert provider.list_models()[0].id == 'model'
        client.close.assert_not_called()
    client.close.assert_called_once()


def test_discovery_cleanup_preserves_primary_failure():
    import pytest
    from app.providers.config import DiscoverySettings
    from app.providers.factory import open_discovery_provider

    client = Mock()
    client.close.side_effect = RuntimeError('private cleanup details')
    failure = ValueError('operation failed')
    with pytest.raises(ValueError) as caught:
        with open_discovery_provider(DiscoverySettings('llama-server', 'http://localhost/v1'),
                                     client_factory=Mock(return_value=client)):
            raise failure
    assert caught.value is failure
    assert any('cleanup' in note for note in failure.__notes__)
    assert all('private' not in note for note in failure.__notes__)
    client.close.assert_called_once()


def test_discovery_cleanup_failure_after_success_is_safe():
    import pytest
    from app.exceptions import ProviderCleanupError
    from app.providers.config import DiscoverySettings
    from app.providers.factory import open_discovery_provider

    client = Mock()
    client.close.side_effect = RuntimeError('private cleanup details')
    with pytest.raises(ProviderCleanupError, match='cleanup') as caught:
        with open_discovery_provider(DiscoverySettings('llama-server', 'http://localhost/v1'),
                                     client_factory=Mock(return_value=client)):
            pass
    assert 'private' not in str(caught.value)


def test_discovery_adapter_does_not_close_borrowed_client():
    from app.providers.config import DiscoverySettings
    from app.providers.openai import OpenAIModelDiscovery

    client = Mock()
    OpenAIModelDiscovery(DiscoverySettings('openai'), client=client).close()
    client.close.assert_not_called()


def test_discovery_rejects_invalid_endpoint_before_construction():
    import pytest
    from app.exceptions import ProviderConfigurationError
    from app.providers.config import DiscoverySettings
    from app.providers.factory import create_discovery_provider

    client_factory = Mock()
    with pytest.raises(ProviderConfigurationError):
        create_discovery_provider(DiscoverySettings('llama-server', 'file:///model'),
                                  client_factory=client_factory)
    client_factory.assert_not_called()


def test_legacy_factory_preserves_combined_remote_adapter():
    from app.providers.remote import LlamaServerProvider

    client = Mock()
    client.models.list.return_value = {'data': [{'id': 'model'}]}
    settings = ProviderSettings('llama-server', 'embedding', 'model', 2,
                                base_url='http://localhost/v1')
    provider = create_provider(settings, client_factory=Mock(return_value=client))
    assert isinstance(provider, LlamaServerProvider)
    assert provider.list_models()[0].id == 'model'
    provider.close()
    client.close.assert_called_once()
