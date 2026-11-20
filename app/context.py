"""Application resources with explicit database connection-pool ownership."""
from app.config import AppConfig, load_config
from dataclasses import dataclass, field

from app.resources import close_resource
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
        close_resource(self.database.close)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        close_resource(self.close, exc)


def create_context(*, config: AppConfig | None = None) -> ApplicationContext:
    config = load_config() if config is None else config
    database = get_database(config)
    try:
        return ApplicationContext(
            documents=database.documents(), storage=get_storage(config), database=database,
        )
    except BaseException as failure:
        close_resource(database.close, failure)
        raise
