from pathlib import Path
from ragworkshop.storage.base import ObjectStorage
from ragworkshop.storage.local import LocalStorage

def get_storage(adapter: str, root: Path) -> ObjectStorage:
    raise NotImplementedError('TODO: get_storage')
