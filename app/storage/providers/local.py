import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from app.exceptions import (
    InvalidStorageKeyError,
    StorageNotFoundError,
    StorageAlreadyExistsError,
    StoragePublicationError,
)
from app.storage.errors import storage_errors
from app.storage.base import ObjectStorage
from app.storage.models import StoredObject


class LocalStorage(ObjectStorage):

    @storage_errors()
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    @storage_errors()
    def _resolve(self, key: str) -> Path:
        if not key or "\\" in key:
            raise InvalidStorageKeyError("Invalid storage key")

        parts = PurePosixPath(key).parts

        if (
            key.startswith("/")
            or any(part in (".", "..") for part in parts)
        ):
            raise InvalidStorageKeyError("Invalid storage key")

        destination = self.root.joinpath(*parts).resolve()

        if not destination.is_relative_to(self.root):
            raise InvalidStorageKeyError("Storage key escapes root")

        if destination == self.root:
            raise InvalidStorageKeyError("Invalid storage key")

        return destination

    def save(
        self,
        source: Path,
        key: str,
        content_type: str,
    ) -> StoredObject:
        source = Path(source)

        if not source.is_file():
            raise FileNotFoundError(source)

        destination = self._resolve(key)
        with storage_errors():
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise StorageAlreadyExistsError(key)

        digest = hashlib.sha256()
        staged = None
        failure = None
        try:
            # Source reads retain native filesystem exceptions; storage writes
            # are translated at their own boundary.
            with source.open('rb') as reader:
                with storage_errors():
                    temp = tempfile.NamedTemporaryFile(
                        dir=destination.parent, prefix='.rypiq-', delete=False,
                    )
                    staged = Path(temp.name)
                read_failure = None
                try:
                    while chunk := reader.read(1024 * 1024):
                        with storage_errors():
                            temp.write(chunk)
                        digest.update(chunk)
                except BaseException as exc:
                    read_failure = exc
                    raise
                finally:
                    try:
                        with storage_errors():
                            temp.close()
                    except Exception:
                        if read_failure is None:
                            raise
                        read_failure.add_note('Could not close the owned staging file.')
            with storage_errors():
                stat = staged.stat()
                result = StoredObject(
                    key=key, size=stat.st_size, content_type=content_type,
                    checksum=digest.hexdigest(),
                    created_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                )
                # Commit point: expose complete data without replacing a writer.
                os.link(staged, destination)
            return result
        except BaseException as exc:
            failure = exc
            raise
        finally:
            if staged is not None:
                try:
                    staged.unlink(missing_ok=True)
                except OSError:
                    if failure is not None:
                        failure.add_note('Could not remove the owned staging file.')
                    else:
                        raise StoragePublicationError(
                            'Object was published, but staging-file cleanup failed',
                            operation='storage:save', completed_steps=('publish',), storage_key=key,
                        ) from None

    @storage_errors()
    def get(
        self,
        key: str,
        destination: Path,
    ) -> Path:
        source = self._resolve(key)

        if not source.is_file():
            raise StorageNotFoundError(key)

        destination = Path(destination)

        if destination.exists():
            raise StorageAlreadyExistsError(destination)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with source.open("rb") as reader:
            with destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer)

        return destination

    @storage_errors()
    def list(
        self,
        prefix: str = "",
    ) -> list[StoredObject]:
        if prefix:
            self._resolve(prefix)

        if not self.root.exists():
            return []

        objects = []

        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue

            key = path.relative_to(self.root).as_posix()

            if not key.startswith(prefix):
                continue

            stat = path.stat()

            objects.append(
                StoredObject(
                    key=key,
                    size=stat.st_size,
                    content_type="application/octet-stream",
                    checksum="",
                    created_at=datetime.fromtimestamp(
                        stat.st_mtime,
                        tz=timezone.utc,
                    ),
                )
            )

        return objects

    @storage_errors()
    def delete(self, key: str) -> bool:
        path = self._resolve(key)

        if not path.is_file():
            return False

        path.unlink()
        return True

    @storage_errors()
    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()
