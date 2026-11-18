from abc import ABC, abstractmethod
from pathlib import Path

from app.storage.models import StoredObject


class ObjectStorage(ABC):
    """Built-in adapters raise StorageError for object I/O and preserve source I/O.

    Invalid keys raise InvalidStorageKeyError. Publication followed by failed
    staging cleanup raises StoragePublicationError with the published key.
    Missing deletes return False. See docs/EXCEPTIONS.md for compatibility.
    """

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
