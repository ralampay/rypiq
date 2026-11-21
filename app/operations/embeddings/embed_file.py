"""Embedding workflows using caller-owned providers."""
import math

from app.logging import log_operation

from app.exceptions import EmptyDocumentError, ProviderResponseError, ValidationError
from app.providers.contracts.embedding import EmbeddingProvider
from app.operations.processing.file import process_text_file

DEFAULT_PREFIX = 'title: none | text: '


def prepare_text_file(path: str, chunk_size: int, stride: int, batch_size: int) -> dict:
    if type(batch_size) is not int or batch_size < 1:
        raise ValidationError('batch_size must be a positive integer')
    document = process_text_file(path=path, chunk_size=chunk_size, stride=stride)
    if not any(chunk['text'].strip() for chunk in document['chunks']):
        raise EmptyDocumentError('File has no nonempty text chunks')
    return document


def embed_chunks(document: dict, provider: EmbeddingProvider, batch_size: int,
                 document_prefix: str = DEFAULT_PREFIX) -> dict:
    """Embed a prepared document; never create or close the borrowed provider."""
    with log_operation('embed') as outcome:
        chunks = [chunk for chunk in document['chunks'] if chunk['text'].strip()]
        records = []
        dimensions = None
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            vectors = provider.embed([document_prefix + chunk['text'] for chunk in batch])
            if not isinstance(vectors, (list, tuple)) or len(vectors) != len(batch):
                raise ProviderResponseError('Embedding count mismatch')
            for chunk, vector in zip(batch, vectors, strict=True):
                if not isinstance(vector, (list, tuple)) or not vector:
                    raise ProviderResponseError('Empty embedding vector')
                if any(isinstance(value, bool) or not isinstance(value, (int, float))
                       or not math.isfinite(value) for value in vector):
                    raise ProviderResponseError('Embedding contains invalid values')
                if dimensions is None:
                    dimensions = len(vector)
                elif len(vector) != dimensions:
                    raise ProviderResponseError('Inconsistent vector dimensions')
                records.append({**chunk, 'embedding': vector})
        outcome['chunk_count'] = len(records)
        return {**document, 'model': provider.model_name, 'dimensions': dimensions,
                'embedded_chunk_count': len(records), 'chunks': records}


class EmbedTextFile:
    def __init__(
        self,
        *,
        path: str,
        chunk_size: int,
        stride: int,
        embedding_provider: EmbeddingProvider,
        batch_size: int = 1,
        document_prefix: str = DEFAULT_PREFIX,
    ) -> None:
        self.document_prefix = document_prefix
        self.path = path
        self.chunk_size = chunk_size
        self.stride = stride
        self.provider = embedding_provider
        self.batch_size = batch_size

    def execute(self) -> dict:
        with log_operation('embed_file'):
            document = prepare_text_file(self.path, self.chunk_size, self.stride, self.batch_size)
            return embed_chunks(document, self.provider, self.batch_size, self.document_prefix)
