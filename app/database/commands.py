"""CLI services reuse the same database initialization as application callers."""
from app.database.dispatch import dispatch
from app.database.factory import get_database


class DatabaseCommands:
    def __init__(self, backend):
        self.backend = backend

    def create(self):
        return self.backend.create_database()

    def migrate(self):
        return self.backend.migrate()

    def drop(self):
        return self.backend.drop_database()

    def reset(self):
        return self.backend.reset()


def run_database_command(args):
    with get_database() as backend:
        return dispatch(args, DatabaseCommands(backend))
