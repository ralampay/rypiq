from abc import ABC, abstractmethod
from pathlib import Path

from app.storage.models import StoredObject


class ObjectStorage(ABC):

    @abstractmethod
    def save(
        self,
        source: Path,
        key: str,
        content_type: str,
    ) -> StoredObject:
        raise NotImplementedError

    @abstractmethod
    def get(
        self,
        key: str,
        destination: Path,
    ) -> Path:
        raise NotImplementedError

    @abstractmethod
    def list(
        self,
        prefix: str = "",
    ) -> list[StoredObject]:
        raise NotImplementedError

    @abstractmethod
    def delete(self, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def exists(self, key: str) -> bool:
        raise NotImplementedError
