from collections.abc import Callable
from dataclasses import dataclass
from ragworkshop.database.base import DatabaseBackend


@dataclass(frozen=True)
class Migration:
    version: str
    apply: Callable[[DatabaseBackend], None]


MIGRATIONS = [Migration("0001", lambda backend: backend.create_documents_collection())]


def migrate(backend: DatabaseBackend) -> list[str]:
    backend.create_database()
    applied = backend.applied_migrations()
    completed = []
    for migration in MIGRATIONS:
        if migration.version not in applied:
            migration.apply(backend)
            backend.record_migration(migration.version)
            completed.append(migration.version)
    return completed
