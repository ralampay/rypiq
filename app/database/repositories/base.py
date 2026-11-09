from abc import ABC, abstractmethod
from uuid import UUID

from app.models.document import Document


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
