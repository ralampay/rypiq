from uuid import UUID

from app.database.repositories.embedding import EmbeddingRepository
from app.services._validation import validate_id


def delete_embedding(*, embedding_id: UUID, repository: EmbeddingRepository) -> bool:
    validate_id(embedding_id)
    return repository.delete(embedding_id)
