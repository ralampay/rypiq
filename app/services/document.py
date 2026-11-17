"""CRUD for pre-embedded documents within one application; never touches storage."""
from dataclasses import replace
import json
from uuid import UUID

from app.database.repositories.base import ApplicationRepository, DocumentRepository
from app.models.document import Document
from app.models.embedding import validate_embedding
from app.services._validation import validate_id, validate_pagination


def _validate_document(document: Document) -> None:
    validate_id(document.id)
    if not isinstance(document.content, str):
        raise ValueError('Document content must be a string')
    if not isinstance(document.metadata, dict):
        raise ValueError('Document metadata must be a JSON object')
    try:
        json.dumps(document.metadata, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError('Document metadata must be a JSON object') from None
    validate_embedding(document.embedding, document.embedding_size)


class DocumentService:
    def __init__(self, *, application_id: UUID, repository: DocumentRepository,
                 applications: ApplicationRepository) -> None:
        self.application_id = application_id
        self.repository = repository
        self.applications = applications

    def create(self, document: Document) -> Document:
        validate_id(self.application_id)
        if document.application_id not in (None, self.application_id):
            raise ValueError('Document belongs to another application')
        application = self.applications.get(self.application_id)
        if application is None:
            raise KeyError(self.application_id)
        owned = replace(document, application_id=application.id,
                        embedding_model=application.embedding_model,
                        embedding_size=application.embedding_size,
                        provision_type=application.provision_type)
        _validate_document(owned)
        return self.repository.create(owned)

    def get(self, document_id: UUID) -> Document:
        validate_id(self.application_id)
        validate_id(document_id)
        document = self.repository.get(document_id, application_id=self.application_id)
        if document is None:
            raise KeyError(document_id)
        return document

    def list(self, *, limit: int = 100, offset: int = 0) -> list[Document]:
        validate_id(self.application_id)
        validate_pagination(limit, offset)
        return self.repository.list(limit=limit, offset=offset, application_id=self.application_id)

    def update(self, document: Document) -> Document:
        validate_id(self.application_id)
        if document.application_id != self.application_id:
            raise ValueError('Document belongs to another application')
        _validate_document(document)
        return self.repository.update(document, application_id=self.application_id)

    def delete(self, document_id: UUID) -> bool:
        validate_id(self.application_id)
        validate_id(document_id)
        return self.repository.delete(document_id, application_id=self.application_id)
