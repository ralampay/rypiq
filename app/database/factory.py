from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config import AppConfig, DatabaseSettings

from app.database.backends.postgresql import PostgreSQLBackend
from app.database.config import load_database_config


def get_database(config: dict | AppConfig | DatabaseSettings | None = None) -> PostgreSQLBackend:
    from app.config import AppConfig, DatabaseSettings
    if isinstance(config, AppConfig):
        config = config.require_database()
    if isinstance(config, DatabaseSettings):
        config = config.as_dict()
    return PostgreSQLBackend(load_database_config() if config is None else config)
