from pathlib import Path
from uuid import uuid4

from app.database.repositories.base import (
    DocumentRepository,
)
from app.storage.base import ObjectStorage
from app.models.document import Document
from app.operations.storage import save_file
from app.operations.embedding import embed_text_file


def ingest_document(
    path: str,
    storage: ObjectStorage,
    repository: DocumentRepository,
    chunk_size: int,
    stride: int,
    **embedding_options,
) -> list[Document]:
    source = Path(path)

    # Generate embeddings before saving the source object.
    result = embed_text_file(
        path=str(source),
        chunk_size=chunk_size,
        stride=stride,
        **embedding_options,
    )

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
                embedding_model=result["model"],
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

    except Exception:
        # Compensate for this operation's successfully
        # written records and uploaded object.
        for document in reversed(documents):
            repository.delete(document.id)

        storage.delete(stored.key)
        raise

    return documents
