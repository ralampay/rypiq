"""Composition and ownership of provider resources."""
from contextlib import contextmanager
from collections.abc import Iterator
import os

from app.embeddings.factory import get_embedding_provider
from app.providers.config import ProviderSettings
from app.providers.contracts import ProviderError, InferenceProvider, EmbeddingProvider
from app.providers.local import LocalInferenceProvider
from app.providers.remote import LlamaServerProvider, OpenAIProvider


def create_provider(settings: ProviderSettings, *, embedding_factory=None, inference_factory=None,
                    client_factory=None) -> EmbeddingProvider | InferenceProvider:
    if settings.capability not in ('embedding', 'inference'):
        raise ValueError('Unsupported provider capability')
    if settings.provider == 'local':
        if settings.capability == 'embedding':
            factory = embedding_factory or get_embedding_provider
            return factory(model_path=settings.model, context_size=settings.context_size,
                           gpu_layers=settings.gpu_layers)
        return (inference_factory or LocalInferenceProvider)(settings)
    if settings.provider not in ('openai', 'llama-server'):
        raise ValueError('Unsupported provider')
    if settings.provider == 'llama-server' and not settings.base_url:
        raise ValueError('llama-server requires an explicit base URL')
    key = os.getenv(settings.api_key_env) if settings.api_key_env else None
    if settings.api_key_env and not key:
        raise ValueError('Provider credential environment variable is missing or empty')
    if settings.provider == 'openai' and not key:
        raise ValueError('OpenAI requires a credential environment variable')
    if client_factory is None:
        from openai import OpenAI
        client_factory = OpenAI
    try:
        client = client_factory(base_url=settings.base_url, api_key=key or 'no-key-required',
                                timeout=settings.timeout, max_retries=0)
    except Exception:
        raise ProviderError('Provider client could not be configured') from None
    adapter = OpenAIProvider if settings.provider == 'openai' else LlamaServerProvider
    return adapter(settings, client=client, owns_client=True)


@contextmanager
def open_provider(settings: ProviderSettings, **dependencies) -> Iterator[EmbeddingProvider | InferenceProvider]:
    provider = create_provider(settings, **dependencies)
    failure = None
    try:
        yield provider
    except BaseException as exc:
        failure = exc
        raise
    finally:
        try:
            provider.close()
        except Exception:
            if failure is not None:
                failure.add_note('Provider cleanup failed; resources may remain open.')
            else:
                raise ProviderError('Operation completed, but provider cleanup failed') from None
