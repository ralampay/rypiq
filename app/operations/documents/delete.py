from uuid import UUID
from app.database.repositories.document import DocumentRepository
from app.models.validation import validate_id


def delete_document(*, document_id: UUID, repository: DocumentRepository) -> bool:
    validate_id(document_id)
    return repository.delete(document_id)
