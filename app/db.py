"""Database-only composition for library callers and search workflows."""
from collections.abc import Iterator
from contextlib import contextmanager

from app.resources import close_resource
from app.database.factory import get_database
from app.database.repositories.document import DocumentRepository
from app.database.repositories.postgresql.document import PostgreSQLDocumentRepository


@contextmanager
def open_document_repository() -> Iterator[DocumentRepository]:
    with get_database() as backend:
        yield backend.documents()


def get_document_repository() -> PostgreSQLDocumentRepository:
    """Compatibility factory. The caller must call repository.close() when done.

    Prefer open_document_repository() for exception-safe resource ownership.
    """
    backend = get_database()
    try:
        return backend.documents()
    except BaseException as failure:
        close_resource(backend.close, failure)
        raise
