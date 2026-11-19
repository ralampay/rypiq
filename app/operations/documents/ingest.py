"""Compatibility pipeline for callers borrowing an already constructed provider."""
from app.models.application import Application
from app.database.repositories.document import DocumentRepository
from app.storage.base import ObjectStorage
from app.models.document import DocumentDetails
from app.providers.contracts.embedding import EmbeddingProvider


class IngestDocument:
    def __init__(self, *, name: str, path: str, storage: ObjectStorage,
                 repository: DocumentRepository, embedding_provider: EmbeddingProvider,
                 chunk_size: int, stride: int, application: Application, batch_size: int = 1) -> None:
        self.name = name
        self.application = application
        self.path = path
        self.storage = storage
        self.repository = repository
        self.provider = embedding_provider
        self.chunk_size = chunk_size
        self.stride = stride
        self.batch_size = batch_size

    def execute(self) -> DocumentDetails:
        from contextlib import nullcontext
        from app.exceptions import ValidationError
        from app.operations.documents.create import CreateDocument
        from app.operations.documents.process import ProcessDocument
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValidationError('batch_size must be a positive integer')
        created = CreateDocument(name=self.name, path=self.path, application=self.application,
            repository=self.repository, storage=self.storage).execute()
        return ProcessDocument(document_id=created.document.id, application=self.application,
            repository=self.repository, storage=self.storage,
            provider_scope=lambda application: nullcontext(self.provider),
            chunk_size=self.chunk_size, stride=self.stride, batch_size=self.batch_size).execute()
