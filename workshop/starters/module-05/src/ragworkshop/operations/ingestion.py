import mimetypes
from pathlib import Path
from uuid import uuid4
from ragworkshop.context import ApplicationContext
from ragworkshop.database.models import Document
from ragworkshop.operations.documents import process_document
from ragworkshop.operations.embedding import embed_chunks
from ragworkshop.processors.chunks import TextChunker

def ingest_document(path: Path, context: ApplicationContext, chunker: TextChunker) -> list[Document]:
    raise NotImplementedError('TODO: ingest_document')
