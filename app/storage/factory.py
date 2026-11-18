from app.exceptions import StorageConfigurationError
from app.storage.base import ObjectStorage
from app.storage.config import load_storage_config
from app.storage.providers.local import LocalStorage


def get_storage(
    config: dict | None = None,
) -> ObjectStorage:
    settings = (
        config
        if config is not None
        else load_storage_config()
    )

    if not isinstance(settings, dict):
        raise StorageConfigurationError('Storage environment must be a mapping')
    adapter = settings.get("adapter")

    if adapter == "local":
        from pathlib import Path
        if not isinstance(settings.get('root'), (str, Path)) or not str(settings['root']).strip():
            raise StorageConfigurationError('Local storage requires a nonempty root path')
        return LocalStorage(settings["root"])

    # Future implementation only
    #
    # if adapter == "s3":
    #     return S3Storage(
    #         bucket=settings["bucket"],
    #         region=settings["region"],
    #         prefix=settings.get("prefix", ""),
    #     )

    raise StorageConfigurationError(
        f"Unsupported storage adapter: {adapter}"
    )
