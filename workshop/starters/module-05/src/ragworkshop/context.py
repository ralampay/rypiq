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
def create_context(settings: Settings, provider: EmbeddingProvider | None=None) -> Iterator[ApplicationContext]:
    raise NotImplementedError('TODO: create_context')
