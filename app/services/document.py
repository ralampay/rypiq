"""CRUD for parent documents within one application; never touches storage."""
from dataclasses import replace
from uuid import UUID

from app.exceptions import ApplicationNotFoundError, DocumentNotFoundError, ValidationError
from app.database.repositories.application import ApplicationRepository
from app.database.repositories.document import DocumentRepository
from app.models.document import Document
from app.models.validation import validate_id, validate_pagination
from app.models.validation import validate_document as _validate_document


class DocumentService:
    def __init__(self, *, application_id: UUID, repository: DocumentRepository,
                 applications: ApplicationRepository) -> None:
        self.application_id = application_id
        self.repository = repository
        self.applications = applications

    def create(self, document: Document) -> Document:
        validate_id(self.application_id)
        if document.application_id not in (None, self.application_id):
            raise ValidationError('Document belongs to another application')
        application = self.applications.get(self.application_id)
        if application is None:
            raise ApplicationNotFoundError(self.application_id, resource_id=self.application_id)
        owned = replace(document, application_id=application.id)
        _validate_document(owned)
        return self.repository.create(owned)

    def get(self, document_id: UUID) -> Document:
        validate_id(self.application_id)
        validate_id(document_id)
        document = self.repository.get(document_id, application_id=self.application_id)
        if document is None:
            raise DocumentNotFoundError(document_id, resource_id=document_id)
        return document

    def list(self, *, limit: int = 100, offset: int = 0) -> list[Document]:
        validate_id(self.application_id)
        validate_pagination(limit, offset)
        return self.repository.list(limit=limit, offset=offset, application_id=self.application_id)

    def update(self, document: Document) -> Document:
        validate_id(self.application_id)
        if document.application_id != self.application_id:
            raise ValidationError('Document belongs to another application')
        _validate_document(document)
        return self.repository.update(document, application_id=self.application_id)

    def delete(self, document_id: UUID) -> bool:
        validate_id(self.application_id)
        validate_id(document_id)
        return self.repository.delete(document_id, application_id=self.application_id)
