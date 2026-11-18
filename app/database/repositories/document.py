from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from app.models.document import Document, DocumentDetails, DocumentSummary
from app.models.embedding import Embedding


class DocumentRepository(ABC):

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
        *, application_id: UUID | None = None,
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
                       offset: int = 0) -> list[DocumentSummary]:
        raise NotImplementedError
