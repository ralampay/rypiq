import mimetypes
from pathlib import Path
from uuid import uuid4

from ragworkshop.context import ApplicationContext
from ragworkshop.database.models import Document
from ragworkshop.operations.documents import process_document
from ragworkshop.operations.embedding import embed_chunks
from ragworkshop.processors.chunks import TextChunker


def ingest_document(path: Path, context: ApplicationContext, chunker: TextChunker) -> list[Document]:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("Input must be a regular file")
    context.documents.ensure_space(context.embeddings.model_name, context.embeddings.dimensions)
    source_id = str(uuid4())
    key = f"documents/{source_id}{source.suffix.lower()}"
    content_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
    stored = context.storage.save(source, key, content_type)
    ids: list[str] = []
    try:
        chunks = process_document(source, chunker)
        if not chunks:
            raise ValueError("Document has no extractable text; scanned PDF may require OCR")
        vectors = embed_chunks(chunks, context.embeddings)
        ids = [str(uuid4()) for chunk in chunks]
        documents = [Document(record_id, source_id, chunk.text, vector, stored.key,
                              {**chunk.metadata, "chunk_index": chunk.index, "start": chunk.start,
                               "end": chunk.end, "checksum": stored.checksum,
                               "embedding_space": context.embeddings.model_name})
                     for record_id, chunk, vector in zip(ids, chunks, vectors, strict=True)]
        context.documents.add(documents)
        return documents
    except Exception as original:
        # Include all allocated IDs: an adapter can fail after a partial write.
        failures = []
        try:
            context.documents.delete(ids)
        except Exception as cleanup:
            failures.append(cleanup)
        try:
            context.storage.delete(key)
        except Exception as cleanup:
            failures.append(cleanup)
        if failures:
            raise ExceptionGroup("Ingestion failed and cleanup was incomplete", [original, *failures]) from original
        raise
