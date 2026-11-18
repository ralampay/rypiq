"""Provider ownership scopes shared by construction entrypoints."""
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

from app.exceptions import ProviderCleanupError
from app.providers.contracts.embedding import EmbeddingProvider
from app.providers.contracts.inference import InferenceProvider
from app.providers.contracts.discovery import ModelDiscovery

Provider = TypeVar('Provider', EmbeddingProvider, InferenceProvider, ModelDiscovery)


@contextmanager
def managed_provider(provider: Provider) -> Iterator[Provider]:
    """Close an owned provider without masking a primary operation failure."""
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
                raise ProviderCleanupError('Operation completed, but provider cleanup failed') from None
