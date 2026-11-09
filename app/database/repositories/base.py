from __future__ import annotations
from abc import ABC, abstractmethod
from uuid import UUID

from app.models.document import Document
from abc import abstractmethod
from app.models.search import SearchResult


class DocumentRepository(ABC):

    @abstractmethod
    def create(self, document: Document) -> Document:
        raise NotImplementedError

    @abstractmethod
    def get(self, id: UUID) -> Document | None:
        raise NotImplementedError

    @abstractmethod
    def update(self, document: Document) -> Document:
        raise NotImplementedError

    @abstractmethod
    def delete(self, id: UUID) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        raise NotImplementedError

    @abstractmethod
    def search_by_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:
        raise NotImplementedError
