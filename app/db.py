"""Database-only composition for library callers and search workflows."""
from collections.abc import Iterator
from contextlib import contextmanager

from app.database.factory import get_database
from app.database.repositories.base import DocumentRepository
from app.database.repositories.postgresql import PostgreSQLDocumentRepository


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
    except Exception:
        backend.close()
        raise
