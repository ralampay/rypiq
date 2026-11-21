"""Rebuild embeddings from a stored original, independently of entrypoints."""
from collections.abc import Callable
from contextlib import AbstractContextManager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import UUID

from app.database.repositories.application import ApplicationRepository
from app.database.repositories.document import DocumentRepository
from app.exceptions import ModelMismatchError, StorageNotFoundError, ValidationError
from app.models.application import Application, chunking_settings, validate_application
from app.models.document import Document, DocumentDetails
from app.models.embedding import Embedding
from app.models.validation import validate_id
from app.operations.applications.get import get_application
# Retain this import for callers of the original helper location.
from app.operations.documents.failure import record_failure
from app.operations.embeddings.build import build_embeddings
from app.operations.embeddings.embed_file import EmbedTextFile
from app.providers.config import embedding_preprocessing, model_identity
from app.providers.contracts.embedding import EmbeddingProvider
from app.storage.base import ObjectStorage


class ProcessDocument:
    def __init__(
        self,
        *,
        document_id: UUID,
        repository: DocumentRepository,
        storage: ObjectStorage,
        provider_scope: Callable[[Application], AbstractContextManager[EmbeddingProvider]],
        applications: ApplicationRepository | None = None,
        application: Application | None = None,
        batch_size: int = 1,
        chunk_size: int | None = None,
        stride: int | None = None,
    ) -> None:
        self.document_id = document_id
        self.repository = repository
        self.storage = storage
        self.provider_scope = provider_scope
        self.applications = applications
        self.application = application
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.stride = stride

    def execute(self) -> DocumentDetails:
        validate_id(self.document_id)
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValidationError('batch_size must be a positive integer')
        with self.repository.processing_lock(self.document_id):
            document = self.repository.set_status(self.document_id, 'processing')
            try:
                application = self._get_application(document)
                chunks = self._build_embeddings(document, application)
                # Release owned model/temp resources before committing success.
                return self.repository.replace_embeddings(
                    document, chunks, application=application,
                )
            except Exception as failure:
                record_failure(self.repository, self.document_id, failure)
                raise

    def _get_application(self, document: Document) -> Application:
        application = self.application
        if self.applications is not None:
            application = get_application(
                application_id=document.application_id,
                repository=self.applications,
            )
        if application is None or application.id != document.application_id:
            raise ValidationError('Processing requires the associated application')
        validate_application(application)
        return application

    def _build_embeddings(
        self, document: Document, application: Application,
    ) -> list[Embedding]:
        size, stride = chunking_settings(application.data)
        size = size if self.chunk_size is None else self.chunk_size
        stride = stride if self.stride is None else self.stride
        if not document.storage_key:
            raise StorageNotFoundError('Document has no stored original')
        with TemporaryDirectory(prefix='rypiq-process-') as temporary:
            source = Path(temporary) / ('source' + Path(document.filename).suffix.lower())
            self.storage.get(document.storage_key, source)
            with self.provider_scope(application) as provider:
                if provider.model_name != model_identity(
                    application.embedding_model, application.embedding_provision_type,
                ):
                    raise ModelMismatchError('Embedding provider model must match the application')
                document_prefix = embedding_preprocessing(
                    application.data,
                    application.embedding_provision_type,
                    application.embedding_size,
                )[0]
                embedded = EmbedTextFile(
                    path=str(source),
                    chunk_size=size,
                    stride=stride,
                    embedding_provider=provider,
                    batch_size=self.batch_size,
                    document_prefix=document_prefix,
                ).execute()
                return build_embeddings(document.id, embedded, application, size, stride)
