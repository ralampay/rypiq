from abc import ABC, abstractmethod

from app.database.repositories.base import (
    DocumentRepository, ApplicationRepository, EmbeddingRepository,
)


class DatabaseBackend(ABC):

    @abstractmethod
    def create_database(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def documents(self) -> DocumentRepository:
        raise NotImplementedError

    @abstractmethod
    def embeddings(self) -> EmbeddingRepository:
        raise NotImplementedError

    @abstractmethod
    def applications(self) -> ApplicationRepository:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
