from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID
from contextlib import AbstractContextManager
from app.models.application import Application

from app.models.document import Document, DocumentDetails, DocumentSummary
from app.models.embedding import Embedding


class DocumentRepository(ABC):
    """Persistence contract; built-in adapters raise app.exceptions types.

    Reads return None when absent; required parents and missing updates raise
    NotFoundError subclasses. Writes may raise ValidationError, ConflictError,
    or DatabaseError. Deleting an absent document returns False.
    """

    @abstractmethod
    def create(self, document: Document) -> Document:
        raise NotImplementedError

    @abstractmethod
    def get(self, id: UUID, *, application_id: UUID | None = None) -> Document | None:
        raise NotImplementedError

    @abstractmethod
    def update(self, document: Document, *, application_id: UUID | None = None) -> Document:
        raise NotImplementedError

    @abstractmethod
    def delete(self, id: UUID, *, application_id: UUID | None = None) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        *, application_id: UUID | None = None, status: str | None = None,
    ) -> list[Document]:
        raise NotImplementedError

    @abstractmethod
    def create_with_embeddings(self, document: Document, embeddings: list[Embedding]) -> DocumentDetails:
        """Insert the complete document atomically, or insert nothing."""
        raise NotImplementedError

    @abstractmethod
    def details(self, id: UUID, *, application_id: UUID | None = None) -> DocumentDetails | None:
        raise NotImplementedError

    @abstractmethod
    def name_exists(self, *, application_id: UUID, name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_summaries(self, *, application_id: UUID, limit: int = 100,
                       offset: int = 0, status: str | None = None) -> list[DocumentSummary]:
        raise NotImplementedError

    @abstractmethod
    def set_status(self, id: UUID, status: str) -> Document:
        """Commit a status change without overwriting other document fields."""
        raise NotImplementedError

    @abstractmethod
    def processing_lock(self, id: UUID) -> AbstractContextManager[None]:
        """Context manager holding an exclusive nonblocking processing lease."""
        raise NotImplementedError

    @abstractmethod
    def replace_embeddings(self, document: Document, embeddings: list[Embedding], *, application: Application) -> DocumentDetails:
        """Atomically replace chunks and mark done, checking captured settings."""
        raise NotImplementedError
