from pathlib import Path
from uuid import uuid4

from app.database.repositories.base import (
    DocumentRepository,
)
from app.storage.base import ObjectStorage
from app.models.document import Document
from app.operations.storage import save_file
from app.operations.embedding import EmbedTextFile, embed_text_file
from app.embeddings.base import EmbeddingProvider


def persist_document(path: str, storage: ObjectStorage,
                     repository: DocumentRepository, chunk_size: int,
                     stride: int, result: dict) -> list[Document]:
    """Persist an embedded file, compensating only for this invocation's writes."""
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


class IngestDocument:
    def __init__(self, *, path: str, storage: ObjectStorage,
                 repository: DocumentRepository, embedding_provider: EmbeddingProvider,
                 chunk_size: int, stride: int, batch_size: int = 1) -> None:
        self.path = path
        self.storage = storage
        self.repository = repository
        self.provider = embedding_provider
        self.chunk_size = chunk_size
        self.stride = stride
        self.batch_size = batch_size

    def execute(self) -> list[Document]:
        result = EmbedTextFile(path=self.path, chunk_size=self.chunk_size,
                               stride=self.stride, embedding_provider=self.provider,
                               batch_size=self.batch_size).execute()
        return persist_document(self.path, self.storage, self.repository,
                                self.chunk_size, self.stride, result)


def ingest_document(path: str, storage: ObjectStorage, repository: DocumentRepository,
                    chunk_size: int, stride: int, **embedding_options) -> list[Document]:
    """Compatibility entrypoint: compose embedding, then use the shared persistence operation."""
    result = embed_text_file(path=path, chunk_size=chunk_size, stride=stride, **embedding_options)
    return persist_document(path, storage, repository, chunk_size, stride, result)
