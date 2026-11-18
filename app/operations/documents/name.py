"""Validate ingestion names before generating embeddings or storing originals."""
from uuid import UUID
from app.exceptions import DuplicateDocumentNameError
from app.database.repositories.document import DocumentRepository
from app.models.validation import normalize_document_name
from app.models.validation import validate_id


def prepare_document_name(*, name: str, application_id: UUID, repository: DocumentRepository) -> str:
    validate_id(application_id)
    name = normalize_document_name(name)
    if repository.name_exists(application_id=application_id, name=name):
        raise DuplicateDocumentNameError('A document with this name already exists in the application', application_id=application_id, name=name)
    return name
