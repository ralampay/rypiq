"""Embedding workflows with caller-supplied providers and no configuration lookup."""
import math
from collections.abc import Callable
from typing import TYPE_CHECKING
from uuid import UUID

from app.database.repositories.base import DocumentRepository
from app.models.document import Document
from app.services._validation import validate_id

from app.embeddings.base import EmbeddingProvider
from app.embeddings.config import DEFAULT_MODEL
from app.operations.text_file import process_text_file

if TYPE_CHECKING:
    from app.providers.session import LocalModelSession

DEFAULT_PREFIX = 'title: none | text: '


def prepare_text_file(path: str, chunk_size: int, stride: int, batch_size: int) -> dict:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    document = process_text_file(path=path, chunk_size=chunk_size, stride=stride)
    if not any(chunk['text'].strip() for chunk in document['chunks']):
        raise ValueError('File has no nonempty text chunks')
    return document


def embed_chunks(document: dict, provider: EmbeddingProvider, batch_size: int,
                 document_prefix: str = DEFAULT_PREFIX) -> dict:
    """Embed a prepared document; never create or close the borrowed provider."""
    chunks = [chunk for chunk in document['chunks'] if chunk['text'].strip()]
    records = []
    dimensions = None
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors = provider.embed([document_prefix + chunk['text'] for chunk in batch])
        if len(vectors) != len(batch):
            raise ValueError('Embedding count mismatch')
        for chunk, vector in zip(batch, vectors, strict=True):
            if not vector:
                raise ValueError('Empty embedding vector')
            if any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) for value in vector):
                raise ValueError('Embedding contains invalid values')
            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise ValueError('Inconsistent vector dimensions')
            records.append({**chunk, 'embedding': vector})
    return {**document, 'model': provider.model_name, 'dimensions': dimensions,
            'embedded_chunk_count': len(records), 'chunks': records}


class EmbedTextFile:
    def __init__(self, *, path: str, chunk_size: int, stride: int,
                 embedding_provider: EmbeddingProvider, batch_size: int = 1,
                 document_prefix: str = DEFAULT_PREFIX) -> None:
        self.document_prefix = document_prefix
        self.path = path
        self.chunk_size = chunk_size
        self.stride = stride
        self.provider = embedding_provider
        self.batch_size = batch_size

    def execute(self) -> dict:
        document = prepare_text_file(self.path, self.chunk_size, self.stride, self.batch_size)
        return embed_chunks(document, self.provider, self.batch_size, self.document_prefix)


def embed_text_file(path: str, chunk_size: int, stride: int,
                    model_path: str = DEFAULT_MODEL, context_size: int | None = None,
                    gpu_layers: int | None = None, batch_size: int = 1,
                    provider_factory: Callable | None = None, document_prefix: str | None = None, *,
                    model_session: 'LocalModelSession | None' = None) -> dict:
    """Compatibility entrypoint; borrow session models or own per-call providers."""
    from app.application import embed_file
    return embed_file(path=path, chunk_size=chunk_size, stride=stride,
                      model_path=model_path, context_size=context_size,
                      gpu_layers=gpu_layers, batch_size=batch_size,
                      provider_factory=provider_factory, document_prefix=document_prefix, model_session=model_session)


def get_embedding(*, embedding_id: UUID, repository: DocumentRepository) -> Document:
    validate_id(embedding_id)
    document = repository.get(embedding_id)
    if document is None:
        raise KeyError(f'Embedding {embedding_id} not found')
    return document


def delete_embedding(*, embedding_id: UUID, repository: DocumentRepository) -> bool:
    validate_id(embedding_id)
    return repository.delete(embedding_id)
