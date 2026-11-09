import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_database_config() -> dict:
    config_file = Path(
        os.getenv(
            "RYPIQ_DATABASE_CONFIG",
            PROJECT_ROOT / "config/database.yml",
        )
    ).expanduser().resolve()

    environment = os.getenv(
        "DATABASE_ENV",
        "development",
    )

    with config_file.open(encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    if not isinstance(settings, dict):
        raise ValueError("Invalid database configuration")

    if environment not in settings:
        raise ValueError(
            f"Unknown database environment: {environment}"
        )

    config = dict(settings[environment])

    if config.get("adapter") == "chromadb":
        path = Path(config["path"]).expanduser()

        if not path.is_absolute():
            path = config_file.parent.parent / path

        config["path"] = str(path.resolve())

    return config
