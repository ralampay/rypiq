from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class StoredObject:
    key: str
    size: int
    content_type: str
    checksum: str


class ObjectStorage(Protocol):
    def save(self, source: Path, key: str, content_type: str) -> StoredObject: ...
    def get(self, key: str, destination: Path) -> Path: ...
    def list(self, prefix: str = "") -> list[StoredObject]: ...
    def delete(self, key: str) -> bool: ...
    def exists(self, key: str) -> bool: ...
