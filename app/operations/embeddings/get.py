from uuid import UUID

from app.exceptions import EmbeddingNotFoundError
from app.database.repositories.embedding import EmbeddingRepository
from app.models.validation import validate_id
from app.models.embedding import Embedding


def get_embedding(*, embedding_id: UUID, repository: EmbeddingRepository) -> Embedding:
    validate_id(embedding_id)
    document = repository.get(embedding_id)
    if document is None:
        raise EmbeddingNotFoundError(f'Embedding {embedding_id} not found', resource_id=embedding_id)
    return document
