from pathlib import Path

from app.models.application import Application, validate_application
from app.providers.config import model_identity
from app.models.embedding import Embedding, validate_embedding
from app.database.repositories.base import DocumentRepository
from app.storage.base import ObjectStorage
from app.models.document import Document, DocumentDetails
from app.operations.storage.save import save_file


def persist_document(path: str, storage: ObjectStorage,
                     repository: DocumentRepository, chunk_size: int,
                     stride: int, result: dict, application: Application) -> DocumentDetails:
    """Commit one aggregate; compensate its original file if the transaction fails."""
    validate_application(application)
    if result['model'] != model_identity(application.embedding_model, application.provision_type):
        raise ValueError('Embedding provider model must match the application')
    if not result['chunks']:
        raise ValueError('File has no nonempty text chunks')
    source = Path(path)
    document = Document(application_id=application.id, filename=source.name, source=str(source))
    chunks = [Embedding(
        document_id=document.id, content=chunk['text'],
        embedding=validate_embedding(chunk['embedding'], application.embedding_size),
        chunk_index=chunk['index'], model_name=result['model'],
        embedding_model=application.embedding_model, embedding_size=application.embedding_size,
        provision_type=application.provision_type,
        metadata={'start': chunk['start'], 'end': chunk['end'],
                  'chunk_size': chunk_size, 'stride': stride},
    ) for chunk in result['chunks']]
    from app.models.validation import validate_chunk
    for chunk in chunks:
        validate_chunk(chunk)
    stored = save_file(path=source, storage=storage)
    document.storage_key = stored.key
    document.content_type = stored.content_type
    document.metadata = {'storage_checksum': stored.checksum}
    try:
        repository.create_with_embeddings(document, chunks)
    except Exception as failure:
        try:
            storage.delete(stored.key)
        except Exception:
            failure.add_note('Could not compensate the stored source object; cleanup is incomplete.')
        raise
    return DocumentDetails(document, chunks)
