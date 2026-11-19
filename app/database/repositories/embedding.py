from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from app.models.embedding import Embedding
from app.models.search import SearchResult


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
                         metadata_filters: dict | None = None, expected_settings: dict | None = None) -> list[SearchResult]:
        raise NotImplementedError
