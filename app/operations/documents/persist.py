from pathlib import Path

from app.exceptions import EmptyDocumentError, ModelMismatchError
from app.models.application import Application, validate_application
from app.providers.config import model_identity
from app.database.repositories.document import DocumentRepository
from app.storage.base import ObjectStorage
from app.models.document import Document, DocumentDetails
from app.operations.storage.save import save_file
from app.models.processing import SNAPSHOT_KEY, processing_settings
from app.models.validation import normalize_document_name
# Retain this import for callers of the original helper location.
from app.operations.embeddings.build import build_embeddings


def persist_document(path: str, storage: ObjectStorage,
                     repository: DocumentRepository, chunk_size: int,
                     stride: int, result: dict, application: Application, *, name: str) -> DocumentDetails:
    """Commit one aggregate; compensate its original file if the transaction fails."""
    name = normalize_document_name(name)
    validate_application(application)
    if result['model'] != model_identity(application.embedding_model, application.embedding_provision_type):
        raise ModelMismatchError('Embedding provider model must match the application')
    if not result['chunks']:
        raise EmptyDocumentError('File has no nonempty text chunks')
    source = Path(path)
    document = Document(name=name, application_id=application.id, filename=source.name, source=str(source))
    chunks = build_embeddings(document.id, result, application, chunk_size, stride)
    stored = save_file(path=source, storage=storage)
    document.status = 'done'
    document.storage_key = stored.key
    document.content_type = stored.content_type
    document.metadata = {'storage_checksum': stored.checksum, SNAPSHOT_KEY: processing_settings(application)}
    try:
        repository.create_with_embeddings(document, chunks)
    except Exception as failure:
        try:
            storage.delete(stored.key)
        except Exception:
            failure.add_note('Could not compensate the stored source object; cleanup is incomplete.')
        raise
    return DocumentDetails(document, chunks)
