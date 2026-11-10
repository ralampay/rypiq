from pathlib import Path
from ragworkshop.database.base import DatabaseBackend


def get_database(adapter: str, path: Path) -> DatabaseBackend:
    if adapter != "chroma":
        raise ValueError(f"Unsupported database adapter: {adapter}")
    from ragworkshop.database.chroma import ChromaDatabaseBackend
    return ChromaDatabaseBackend(path)
