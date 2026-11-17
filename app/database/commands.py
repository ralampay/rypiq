"""Database use cases with both injected and configured Python entrypoints."""
from app.database.backends.postgresql import PostgreSQLBackend
from app.database.dispatch import method_for
from app.database.factory import get_database


class DatabaseCommands:
    """Borrow the caller's backend; do not construct or close it here."""
    def __init__(self, backend: PostgreSQLBackend) -> None:
        self.backend = backend

    def create(self) -> dict:
        return self.backend.create_database()

    def migrate(self) -> dict:
        return self.backend.migrate()

    def drop(self) -> dict:
        return self.backend.drop_database()

    def reset(self) -> dict:
        return self.backend.reset()


def execute_database_command(command: str) -> dict:
    method = method_for(command)
    with get_database() as backend:
        return getattr(DatabaseCommands(backend), method)()


def run_database_command(args) -> dict:
    """Compatibility adapter; new Python callers pass a command string instead."""
    return execute_database_command(args.command)
