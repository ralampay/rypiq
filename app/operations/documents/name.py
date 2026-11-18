"""Validate ingestion names before generating embeddings or storing originals."""
from uuid import UUID
from app.database.repositories.base import DocumentRepository
from app.models.validation import normalize_document_name
from app.services._validation import validate_id


def prepare_document_name(*, name: str, application_id: UUID, repository: DocumentRepository) -> str:
    validate_id(application_id)
    name = normalize_document_name(name)
    if repository.name_exists(application_id=application_id, name=name):
        raise ValueError('A document with this name already exists in the application')
    return name
