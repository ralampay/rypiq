from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator

from ragworkshop.config import Settings
from ragworkshop.database.base import DocumentRepository
from ragworkshop.database.factory import get_database
from ragworkshop.database.migrations import migrate
from ragworkshop.embeddings.base import EmbeddingProvider
from ragworkshop.embeddings.factory import get_embedding_provider
from ragworkshop.storage.base import ObjectStorage
from ragworkshop.storage.factory import get_storage


@dataclass(frozen=True)
class ApplicationContext:
    documents: DocumentRepository
    storage: ObjectStorage
    embeddings: EmbeddingProvider


@contextmanager
def create_context(settings: Settings, provider: EmbeddingProvider | None = None) -> Iterator[ApplicationContext]:
    owned = provider is None
    if provider is None:
        provider = get_embedding_provider(settings.embedding_provider, settings.embedding_model,
                                          settings.embedding_dimensions)
    try:
        backend = get_database(settings.database_adapter, settings.database_path)
        migrate(backend)
        repository = backend.documents()
        repository.ensure_space(provider.model_name, provider.dimensions)
        yield ApplicationContext(repository, get_storage(settings.storage_adapter, settings.storage_root), provider)
    finally:
        if owned and hasattr(provider, "close"):
            provider.close()
    # PersistentClient exposes no public close in the pinned Chroma version.
    # CLI subprocess exit releases its resources; never call private reset APIs.
