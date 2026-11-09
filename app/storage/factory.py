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

    adapter = settings.get("adapter")

    if adapter == "local":
        return LocalStorage(settings["root"])

    # Future implementation only
    #
    # if adapter == "s3":
    #     return S3Storage(
    #         bucket=settings["bucket"],
    #         region=settings["region"],
    #         prefix=settings.get("prefix", ""),
    #     )

    raise ValueError(
        f"Unsupported storage adapter: {adapter}"
    )
