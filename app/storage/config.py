import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_storage_config() -> dict:
    config_file = Path(
        os.getenv(
            "RYPIQ_STORAGE_CONFIG",
            PROJECT_ROOT / "config/storage.yml",
        )
    ).expanduser().resolve()

    environment = os.getenv(
        "STORAGE_ENV",
        os.getenv("DATABASE_ENV", "development"),
    )

    with config_file.open(encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    if not isinstance(settings, dict):
        raise ValueError("Invalid storage configuration")

    if environment not in settings:
        raise ValueError(
            f"Unknown storage environment: {environment}"
        )

    config = dict(settings[environment])

    if config.get("adapter") == "local":
        root = Path(config["root"]).expanduser()

        if not root.is_absolute():
            root = config_file.parent.parent / root

        config["root"] = str(root.resolve())

    return config
