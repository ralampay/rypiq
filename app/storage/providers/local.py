import hashlib
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from app.storage.base import ObjectStorage
from app.storage.models import StoredObject


class LocalStorage(ObjectStorage):

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()

    def _resolve(self, key: str) -> Path:
        if not key or "\\" in key:
            raise ValueError("Invalid storage key")

        parts = PurePosixPath(key).parts

        if (
            key.startswith("/")
            or any(part in (".", "..") for part in parts)
        ):
            raise ValueError("Invalid storage key")

        destination = self.root.joinpath(*parts).resolve()

        if not destination.is_relative_to(self.root):
            raise ValueError("Storage key escapes root")

        if destination == self.root:
            raise ValueError("Invalid storage key")

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
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        if destination.exists():
            raise FileExistsError(key)

        digest = hashlib.sha256()
        temp_name = None

        try:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent,
                delete=False,
            ) as temp:
                temp_name = temp.name

                with source.open("rb") as reader:
                    while chunk := reader.read(1024 * 1024):
                        temp.write(chunk)
                        digest.update(chunk)

            # Exclusive creation avoids silently overwriting
            # an existing object with the same key.
            with open(destination, "xb") as output:
                with open(temp_name, "rb") as reader:
                    shutil.copyfileobj(reader, output)

        except Exception:
            if destination.exists():
                # Never delete a pre-existing object on failure.
                # The temp file is cleaned up below.
                pass
            raise

        finally:
            if temp_name is not None:
                Path(temp_name).unlink(missing_ok=True)

        stat = destination.stat()

        return StoredObject(
            key=key,
            size=stat.st_size,
            content_type=content_type,
            checksum=digest.hexdigest(),
            created_at=datetime.fromtimestamp(
                stat.st_mtime,
                tz=timezone.utc,
            ),
        )

    def get(
        self,
        key: str,
        destination: Path,
    ) -> Path:
        source = self._resolve(key)

        if not source.is_file():
            raise FileNotFoundError(key)

        destination = Path(destination)

        if destination.exists():
            raise FileExistsError(destination)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with source.open("rb") as reader:
            with destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer)

        return destination

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

    def delete(self, key: str) -> bool:
        path = self._resolve(key)

        if not path.is_file():
            return False

        path.unlink()
        return True

    def exists(self, key: str) -> bool:
        return self._resolve(key).is_file()
