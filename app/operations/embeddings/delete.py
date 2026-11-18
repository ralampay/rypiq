from uuid import UUID

from app.database.repositories.base import DocumentRepository
from app.services._validation import validate_id


def delete_embedding(*, embedding_id: UUID, repository: DocumentRepository) -> bool:
    validate_id(embedding_id)
    return repository.delete(embedding_id)
