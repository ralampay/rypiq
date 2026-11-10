import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    environment: str = "development"
    database_adapter: str = "chroma"
    database_path: Path = Path(".data/development/chroma")
    storage_adapter: str = "local"
    storage_root: Path = Path(".data/development/objects")
    embedding_provider: str = "fake"
    embedding_model: str = ""
    embedding_dimensions: int = 8


def read_section(path: Path, environment: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError("Invalid YAML configuration") from exc
    if not isinstance(data, dict) or not isinstance(data.get(environment), dict):
        raise ValueError("Missing environment configuration section")
    return data[environment]


def load_settings(overrides: dict[str, Any] | None = None) -> Settings:
    """Call at an entry point only; never from business operations."""
    values = overrides or {}
    load_dotenv(os.getenv("RAG_WORKSHOP_DOTENV", ".env"), override=False)
    def choose(key, default):
        if key in values and values[key] is not None:
            return values[key]
        return os.getenv("RAG_WORKSHOP_" + key.upper(), default)
    environment = choose("env", "development")
    if environment not in {"development", "test", "production"}:
        raise ValueError("Unknown application environment")
    database_file = Path(choose("database_config", "config/database.yml")).expanduser().resolve()
    storage_file = Path(choose("storage_config", "config/storage.yml")).expanduser().resolve()
    database = read_section(database_file, environment)
    storage = read_section(storage_file, environment)
    def resolve(key, section, field, config_file, default):
        explicit = key in values or "RAG_WORKSHOP_" + key.upper() in os.environ
        raw = choose(key, section.get(field, default))
        if not isinstance(raw, (str, Path)) or not str(raw).strip():
            raise ValueError(f"Invalid path setting: {key}")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() if explicit else config_file.parent) / path
        return path.resolve()
    embedding = database.get("embedding", {})
    if not isinstance(embedding, dict):
        raise ValueError("embedding must be a mapping")
    dimensions = int(choose("embedding_dimensions", embedding.get("dimensions", 8)))
    if dimensions < 1:
        raise ValueError("Embedding dimensions must be positive")
    return Settings(
        environment=environment,
        database_adapter=choose("database_adapter", database.get("adapter", "chroma")),
        database_path=resolve("database_path", database, "path", database_file,
                              f"../.data/{environment}/chroma"),
        storage_adapter=choose("storage_adapter", storage.get("adapter", "local")),
        storage_root=resolve("storage_root", storage, "root", storage_file,
                             f"../.data/{environment}/objects"),
        embedding_provider=choose("embedding_provider", embedding.get("provider", "fake")),
        embedding_model=choose("embedding_model", embedding.get("model", "")),
        embedding_dimensions=dimensions,
    )
