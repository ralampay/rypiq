from uuid import UUID
from app.database.repositories.document import DocumentRepository
from app.models.document import DocumentDetails
from app.services._validation import validate_id


def get_document(*, document_id: UUID, repository: DocumentRepository) -> DocumentDetails:
    validate_id(document_id)
    result = repository.details(document_id)
    if result is None:
        raise KeyError(f'Document {document_id} not found')
    return result
