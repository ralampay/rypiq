from app.database.base import DatabaseBackend
from app.database.config import load_database_config


def get_database(
    config: dict | None = None,
) -> DatabaseBackend:

    settings = (
        config
        if config is not None
        else load_database_config()
    )

    adapter = settings.get("adapter")

    if adapter == "chromadb":
        from app.database.backends.chroma import ChromaBackend

        return ChromaBackend(settings)

    raise ValueError(
        f"Unsupported database adapter: {adapter}"
    )
