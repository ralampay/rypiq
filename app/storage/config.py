import os
from pathlib import Path

import yaml
from app.exceptions import StorageConfigurationError


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

    try:
        with config_file.open(encoding="utf-8") as file:
            settings = yaml.safe_load(file)
    except (OSError, UnicodeError, yaml.YAMLError):
        raise StorageConfigurationError('Cannot read storage configuration; check the path and YAML syntax') from None

    if not isinstance(settings, dict):
        raise StorageConfigurationError("Invalid storage configuration")

    if environment not in settings:
        raise StorageConfigurationError(
            f"Unknown storage environment: {environment}"
        )

    if not isinstance(settings[environment], dict):
        raise StorageConfigurationError('Storage environment must be a mapping')
    config = dict(settings[environment])

    if config.get("adapter") == "local":
        if not isinstance(config.get('root'), str) or not config['root'].strip():
            raise StorageConfigurationError('Local storage requires a nonempty root path')
        root = Path(config["root"]).expanduser()

        if not root.is_absolute():
            root = config_file.parent.parent / root

        config["root"] = str(root.resolve())

    return config
