"""Typed provider construction; environment access stays at this boundary."""
from collections.abc import Iterator
from contextlib import contextmanager
import os

from app.exceptions import ProviderConfigurationError, ProviderInitializationError
from app.embeddings.factory import get_embedding_provider
from app.providers.config import ProviderSettings, DiscoverySettings, validate_options
from app.providers.contracts.embedding import EmbeddingProvider
from app.providers.contracts.inference import InferenceProvider
from app.providers.contracts.discovery import ModelDiscovery
from app.providers.local.inference import LocalInferenceProvider
from app.providers.openai import OpenAIEmbeddingProvider, OpenAIInferenceProvider, OpenAIModelDiscovery
from app.providers.llama_server import LlamaServerEmbeddingProvider, LlamaServerInferenceProvider, LlamaServerModelDiscovery
from app.providers.lifecycle import managed_provider
from app.providers.remote import OpenAIProvider, LlamaServerProvider


def _create_client(settings: ProviderSettings | DiscoverySettings, client_factory=None):
    if settings.provider not in ('openai', 'llama-server'):
        raise ProviderConfigurationError('Unsupported provider')
    if settings.provider == 'llama-server' and not settings.base_url:
        raise ProviderConfigurationError('llama-server requires an explicit base URL')
    key = os.getenv(settings.api_key_env) if settings.api_key_env else None
    if settings.api_key_env and not key:
        raise ProviderConfigurationError('Provider credential environment variable is missing or empty')
    if settings.provider == 'openai' and not key:
        raise ProviderConfigurationError('OpenAI requires a credential environment variable')
    try:
        if client_factory is None:
            from openai import OpenAI
            client_factory = OpenAI
        client = client_factory(base_url=settings.base_url, api_key=key or 'no-key-required',
                                timeout=settings.timeout, max_retries=0)
    except Exception:
        raise ProviderInitializationError('Provider client could not be configured') from None
    return client


def create_embedding_provider(settings: ProviderSettings, *, embedding_factory=None,
                              client_factory=None) -> EmbeddingProvider:
    if settings.capability != 'embedding':
        raise ProviderConfigurationError('Provider was not configured for embeddings')
    if settings.provider == 'local':
        factory = embedding_factory or get_embedding_provider
        return factory(model_path=settings.model, context_size=settings.context_size,
                       gpu_layers=settings.gpu_layers)
    client = _create_client(settings, client_factory)
    adapter = OpenAIEmbeddingProvider if settings.provider == 'openai' else LlamaServerEmbeddingProvider
    return adapter(settings, client=client, owns_client=True)


def create_inference_provider(settings: ProviderSettings, *, inference_factory=None,
                              client_factory=None) -> InferenceProvider:
    if settings.capability != 'inference':
        raise ProviderConfigurationError('Provider was not configured for inference')
    if settings.provider == 'local':
        return (inference_factory or LocalInferenceProvider)(settings)
    client = _create_client(settings, client_factory)
    adapter = OpenAIInferenceProvider if settings.provider == 'openai' else LlamaServerInferenceProvider
    return adapter(settings, client=client, owns_client=True)


def create_discovery_provider(settings: DiscoverySettings, *, client_factory=None) -> ModelDiscovery:
    options = {'timeout': settings.timeout}
    if settings.base_url is not None:
        options['base_url'] = settings.base_url
    if settings.api_key_env is not None:
        options['api_key_env'] = settings.api_key_env
    validate_options(options, 'inference')
    client = _create_client(settings, client_factory)
    adapter = OpenAIModelDiscovery if settings.provider == 'openai' else LlamaServerModelDiscovery
    return adapter(settings, client=client, owns_client=True)


def _create_capability_provider(settings: ProviderSettings, *, embedding_factory=None, inference_factory=None,
                    client_factory=None) -> EmbeddingProvider | InferenceProvider:
    """Dispatch to a focused capability adapter."""
    if settings.capability == 'embedding':
        return create_embedding_provider(settings, embedding_factory=embedding_factory,
                                         client_factory=client_factory)
    if settings.capability == 'inference':
        return create_inference_provider(settings, inference_factory=inference_factory,
                                         client_factory=client_factory)
    raise ProviderConfigurationError('Unsupported provider capability')


def create_provider(settings: ProviderSettings, *, embedding_factory=None, inference_factory=None,
                    client_factory=None) -> EmbeddingProvider | InferenceProvider:
    """Legacy dispatcher retaining combined remote adapters and injection hooks."""
    if settings.capability not in ('embedding', 'inference'):
        raise ProviderConfigurationError('Unsupported provider capability')
    if settings.provider == 'local':
        return _create_capability_provider(settings, embedding_factory=embedding_factory,
                                           inference_factory=inference_factory, client_factory=client_factory)
    client = _create_client(settings, client_factory)
    adapter = OpenAIProvider if settings.provider == 'openai' else LlamaServerProvider
    return adapter(settings, client=client, owns_client=True)


@contextmanager
def open_capability_provider(settings: ProviderSettings, **dependencies) -> Iterator[EmbeddingProvider | InferenceProvider]:
    """Own only the requested capability; discovery has its own scope."""
    with managed_provider(_create_capability_provider(settings, **dependencies)) as provider:
        yield provider


@contextmanager
def open_provider(settings: ProviderSettings, **dependencies) -> Iterator[EmbeddingProvider | InferenceProvider]:
    with managed_provider(create_provider(settings, **dependencies)) as provider:
        yield provider


@contextmanager
def open_discovery_provider(settings: DiscoverySettings, **dependencies) -> Iterator[ModelDiscovery]:
    with managed_provider(create_discovery_provider(settings, **dependencies)) as provider:
        yield provider
