"""Application-scoped chunk CRUD using caller-owned resources."""
from uuid import UUID
from app.database.repositories.base import EmbeddingRepository
from app.models.embedding import Embedding
from app.models.validation import validate_chunk
from app.services._validation import validate_id


class EmbeddingService:
    def __init__(self, *, application_id: UUID, repository: EmbeddingRepository):
        self.application_id = application_id
        self.repository = repository

    def create(self, embedding: Embedding) -> Embedding:
        validate_id(self.application_id)
        validate_chunk(embedding)
        return self.repository.create(embedding, application_id=self.application_id)

    def get(self, embedding_id: UUID) -> Embedding:
        validate_id(self.application_id)
        validate_id(embedding_id)
        value = self.repository.get(embedding_id, application_id=self.application_id)
        if value is None:
            raise KeyError(embedding_id)
        return value

    def list(self, *, document_id: UUID) -> list[Embedding]:
        validate_id(self.application_id)
        validate_id(document_id)
        return self.repository.list(document_id=document_id, application_id=self.application_id)

    def update(self, embedding: Embedding) -> Embedding:
        validate_id(self.application_id)
        validate_chunk(embedding)
        return self.repository.update(embedding, application_id=self.application_id)

    def delete(self, embedding_id: UUID) -> bool:
        validate_id(self.application_id)
        validate_id(embedding_id)
        return self.repository.delete(embedding_id, application_id=self.application_id)
