from __future__ import annotations
from abc import ABC, abstractmethod
from uuid import UUID

from app.models.document import Document, DocumentDetails
from app.models.embedding import Embedding
from app.models.application import Application
from app.models.search import SearchResult


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


class EmbeddingRepository(ABC):
    @abstractmethod
    def create(self, embedding: Embedding, *, application_id: UUID | None = None) -> Embedding:
        raise NotImplementedError

    @abstractmethod
    def get(self, id: UUID, *, application_id: UUID | None = None) -> Embedding | None:
        raise NotImplementedError

    @abstractmethod
    def update(self, embedding: Embedding, *, application_id: UUID | None = None) -> Embedding:
        raise NotImplementedError

    @abstractmethod
    def delete(self, id: UUID, *, application_id: UUID | None = None) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list(self, *, document_id: UUID, application_id: UUID | None = None) -> list[Embedding]:
        raise NotImplementedError

    @abstractmethod
    def search_by_vector(self, embedding: list[float], limit: int = 10,
                         filters: dict | None = None, *, application_id: UUID,
                         metadata_filters: dict | None = None) -> list[SearchResult]:
        raise NotImplementedError


class ApplicationRepository(ABC):
    @abstractmethod
    def create(self, application: Application) -> Application:
        raise NotImplementedError

    @abstractmethod
    def get(self, id: UUID) -> Application | None:
        raise NotImplementedError

    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0) -> list[Application]:
        raise NotImplementedError

    @abstractmethod
    def update(self, application: Application) -> Application:
        raise NotImplementedError

    @abstractmethod
    def delete(self, id: UUID) -> bool:
        raise NotImplementedError
