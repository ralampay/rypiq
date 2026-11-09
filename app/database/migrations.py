from dataclasses import dataclass
from typing import Callable

from app.database.base import DatabaseBackend


@dataclass(frozen=True)
class Migration:
    version: str
    description: str
    apply: Callable[[DatabaseBackend], None]


MIGRATIONS = [
    Migration(
        version="0001",
        description="Create documents collection",
        apply=lambda db: db.create_documents_collection(),
    ),
]


def migrate(database: DatabaseBackend) -> list[str]:
    applied = database.applied_migrations()
    completed = []

    for migration in sorted(
        MIGRATIONS,
        key=lambda item: item.version,
    ):
        if migration.version in applied:
            continue

        migration.apply(database)

        database.record_migration(
            migration.version,
            migration.description,
        )

        completed.append(migration.version)

    return completed
