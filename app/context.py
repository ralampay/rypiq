"""Application resources with explicit database connection-pool ownership."""
from dataclasses import dataclass, field

from app.database.base import DatabaseBackend
from app.database.factory import get_database
from app.database.repositories.document import DocumentRepository
from app.database.repositories.application import ApplicationRepository
from app.database.repositories.embedding import EmbeddingRepository
from app.storage.base import ObjectStorage
from app.storage.factory import get_storage


@dataclass
class ApplicationContext:
    documents: DocumentRepository
    storage: ObjectStorage
    database: DatabaseBackend = field(repr=False)

    @property
    def embeddings(self) -> EmbeddingRepository:
        return self.database.embeddings()

    @property
    def applications(self) -> ApplicationRepository:
        return self.database.applications()

    def close(self):
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def create_context() -> ApplicationContext:
    database = get_database()
    try:
        return ApplicationContext(
            documents=database.documents(), storage=get_storage(), database=database,
        )
    except Exception:
        database.close()
        raise
