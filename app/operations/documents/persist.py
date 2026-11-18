from pathlib import Path
from uuid import uuid4

from app.models.application import Application, validate_application
from app.providers.config import model_identity
from app.models.embedding import validate_embedding
from app.database.repositories.base import DocumentRepository
from app.storage.base import ObjectStorage
from app.models.document import Document
from app.operations.storage.save import save_file


def persist_document(path: str, storage: ObjectStorage,
                     repository: DocumentRepository, chunk_size: int,
                     stride: int, result: dict, application: Application) -> list[Document]:
    """Persist an embedded file, compensating only for this invocation's writes."""
    validate_application(application)
    if result['model'] != model_identity(application.embedding_model, application.provision_type):
        raise ValueError('Embedding provider model must match the application')
    for chunk in result['chunks']:
        validate_embedding(chunk['embedding'], application.embedding_size)
    source = Path(path)
    stored = save_file(
        path=source,
        storage=storage,
    )

    source_id = uuid4()
    documents = []

    try:
        for chunk in result["chunks"]:
            document = Document(
                application_id=application.id,
                embedding_model=application.embedding_model,
                embedding_size=application.embedding_size,
                provision_type=application.provision_type,
                content=chunk["text"],
                embedding=chunk["embedding"],
                storage_key=stored.key,
                filename=source.name,
                content_type=stored.content_type,
                source=str(source),
                source_id=source_id,
                chunk_index=chunk["index"],
                model_name=result["model"],
                metadata={
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "chunk_size": chunk_size,
                    "stride": stride,
                    "storage_checksum": stored.checksum,
                },
            )

            repository.create(document)
            documents.append(document)

    except Exception as failure:
        # Compensate for this operation's successfully
        # written records and uploaded object.
        for document in reversed(documents):
            try:
                repository.delete(document.id)
            except Exception:
                failure.add_note(f'Could not compensate document {document.id}; cleanup is incomplete.')

        try:
            storage.delete(stored.key)
        except Exception:
            failure.add_note('Could not compensate the stored source object; cleanup is incomplete.')
        raise

    return documents
