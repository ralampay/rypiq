from app.storage.base import ObjectStorage
from app.storage.models import StoredObject


def list_files(
    storage: ObjectStorage,
    prefix: str = "",
) -> list[StoredObject]:
    return storage.list(prefix=prefix)
