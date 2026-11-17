from app.database.backends.postgresql import PostgreSQLBackend
from app.database.config import load_database_config


def get_database(config=None):
    return PostgreSQLBackend(load_database_config() if config is None else config)
