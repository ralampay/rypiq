from uuid import UUID

from app.database.repositories.base import EmbeddingRepository
from app.services._validation import validate_id
from app.models.embedding import Embedding


def get_embedding(*, embedding_id: UUID, repository: EmbeddingRepository) -> Embedding:
    validate_id(embedding_id)
    document = repository.get(embedding_id)
    if document is None:
        raise KeyError(f'Embedding {embedding_id} not found')
    return document
