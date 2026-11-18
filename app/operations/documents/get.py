from uuid import UUID
from app.exceptions import DocumentNotFoundError
from app.database.repositories.document import DocumentRepository
from app.models.document import DocumentDetails
from app.models.validation import validate_id


def get_document(*, document_id: UUID, repository: DocumentRepository) -> DocumentDetails:
    validate_id(document_id)
    result = repository.details(document_id)
    if result is None:
        raise DocumentNotFoundError(f'Document {document_id} not found', resource_id=document_id)
    return result
