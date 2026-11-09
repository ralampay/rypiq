import mimetypes
from pathlib import Path
from uuid import uuid4

from app.storage.base import ObjectStorage
from app.storage.models import StoredObject


def save_file(
    path: str | Path,
    storage: ObjectStorage,
    prefix: str = "documents",
) -> StoredObject:
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(source)

    # Retain the extension while preventing filename collisions.
    key = f"{prefix.rstrip('/')}/{uuid4()}{source.suffix.lower()}"

    content_type = (
        mimetypes.guess_type(source.name)[0]
        or "application/octet-stream"
    )

    return storage.save(
        source=source,
        key=key,
        content_type=content_type,
    )


def list_files(
    storage: ObjectStorage,
    prefix: str = "",
) -> list[StoredObject]:
    return storage.list(prefix=prefix)
