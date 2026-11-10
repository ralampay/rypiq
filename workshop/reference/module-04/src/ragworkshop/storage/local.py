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
        if (not key or "\\" in key or "\x00" in key or key.startswith("/")
                or any(part in ("", ".", "..") for part in key.split("/"))):
            raise ValueError("Unsafe storage key")
        # A digest gives every key one independent bundle, avoiding prefix collisions.
        return self.root / hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _safe(self, path: Path) -> None:
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Unsafe storage symlink")

    def _object(self, bundle: Path) -> StoredObject:
        self._safe(bundle)
        self._safe(bundle / "metadata.json")
        self._safe(bundle / "payload")
        return StoredObject(**json.loads((bundle / "metadata.json").read_text(encoding="utf-8")))

    def save(self, source: Path, key: str, content_type: str) -> StoredObject:
        bundle = self._path(key)
        self._safe(bundle)
        source = Path(source).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("Source must be a regular file")
        if bundle.exists():
            raise FileExistsError(key)
        if not content_type.strip():
            raise ValueError("Content type must not be empty")
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
        try:
            payload = staging / "payload"
            shutil.copyfile(source, payload)
            with payload.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            stored = StoredObject(key, payload.stat().st_size, content_type, checksum)
            (staging / "metadata.json").write_text(json.dumps(asdict(stored)), encoding="utf-8")
            # Completed bundles are nonempty directories: rename cannot replace one.
            os.rename(staging, bundle)
            return stored
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def get(self, key: str, destination: Path) -> Path:
        bundle = self._path(key)
        self._object(bundle)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            with destination.open("xb") as writer:
                created = True
                with (bundle / "payload").open("rb") as reader:
                    shutil.copyfileobj(reader, writer)
        except Exception:
            if created:
                destination.unlink(missing_ok=True)
            raise
        return destination

    def list(self, prefix: str = "") -> list[StoredObject]:
        if prefix:
            self._path(prefix.rstrip("/"))
        objects = [self._object(bundle) for bundle in self.root.iterdir()
                   if not bundle.name.startswith(".pending-")]
        return sorted((obj for obj in objects if obj.key.startswith(prefix)), key=lambda obj: obj.key)

    def exists(self, key: str) -> bool:
        bundle = self._path(key)
        self._safe(bundle)
        if not bundle.exists():
            return False
        self._object(bundle)
        return True

    def delete(self, key: str) -> bool:
        if not self.exists(key):
            return False
        shutil.rmtree(self._path(key))
        return True
