from dataclasses import dataclass

from app.database.factory import get_database
from app.database.repositories.base import (
    DocumentRepository,
)
from app.storage.base import ObjectStorage
from app.storage.factory import get_storage


@dataclass
class ApplicationContext:
    documents: DocumentRepository
    storage: ObjectStorage


def create_context() -> ApplicationContext:
    return ApplicationContext(
        documents=get_database().documents(),
        storage=get_storage(),
    )
