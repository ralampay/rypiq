"""Database-only composition for library callers and search workflows."""
from app.config import AppConfig, load_config
from collections.abc import Iterator
from contextlib import contextmanager

from app.resources import close_resource
from app.database.factory import get_database
from app.database.repositories.document import DocumentRepository
from app.database.repositories.postgresql.document import PostgreSQLDocumentRepository


@contextmanager
def open_document_repository(*, config: AppConfig | None = None) -> Iterator[DocumentRepository]:
    config = load_config() if config is None else config
    with get_database(config) as backend:
        yield backend.documents()


def get_document_repository(*, config: AppConfig | None = None) -> PostgreSQLDocumentRepository:
    """Compatibility factory. The caller must call repository.close() when done.

    Prefer open_document_repository() for exception-safe resource ownership.
    """
    config = load_config() if config is None else config
    backend = get_database(config)
    try:
        return backend.documents()
    except BaseException as failure:
        close_resource(backend.close, failure)
        raise
