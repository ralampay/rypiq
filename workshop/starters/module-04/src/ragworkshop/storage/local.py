import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from ragworkshop.storage.base import StoredObject

class LocalStorage:
    """Single-process adapter; root must not be writable by untrusted users."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        raise NotImplementedError('TODO: LocalStorage._path')

    def _safe(self, path: Path) -> None:
        raise NotImplementedError('TODO: LocalStorage._safe')

    def _object(self, bundle: Path) -> StoredObject:
        raise NotImplementedError('TODO: LocalStorage._object')

    def save(self, source: Path, key: str, content_type: str) -> StoredObject:
        raise NotImplementedError('TODO: LocalStorage.save')

    def get(self, key: str, destination: Path) -> Path:
        raise NotImplementedError('TODO: LocalStorage.get')

    def list(self, prefix: str='') -> list[StoredObject]:
        raise NotImplementedError('TODO: LocalStorage.list')

    def exists(self, key: str) -> bool:
        raise NotImplementedError('TODO: LocalStorage.exists')

    def delete(self, key: str) -> bool:
        raise NotImplementedError('TODO: LocalStorage.delete')
