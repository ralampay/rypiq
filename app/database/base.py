from abc import ABC, abstractmethod

from app.database.repositories.base import (
    DocumentRepository,
)


class DatabaseBackend(ABC):

    @abstractmethod
    def create_database(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def create_documents_collection(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def applied_migrations(self) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def record_migration(
        self,
        version: str,
        description: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def documents(self) -> DocumentRepository:
        raise NotImplementedError
