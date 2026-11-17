"""Configured Python entrypoints. Own resources here; keep commands injectable."""
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from app.context import create_context
from app.db import open_document_repository
from app.embeddings.base import EmbeddingProvider
from app.embeddings.config import DEFAULT_MODEL
from app.embeddings.factory import get_embedding_provider
from app.models.document import Document
from app.models.search import SearchResult
from app.operations.document import IngestDocument
from app.operations.embedding import embed_chunks, prepare_text_file
from app.operations.search import SearchDocuments


@contextmanager
def open_embedding_provider(*, model_path: str = DEFAULT_MODEL, context_size: int = 2048,
                            gpu_layers: int = 0, provider_factory: Callable | None = None
                            ) -> Iterator[EmbeddingProvider]:
    if type(context_size) is not int or context_size < 1:
        raise ValueError('context_size must be positive')
    factory = provider_factory if provider_factory is not None else get_embedding_provider
    provider = factory(model_path=model_path, context_size=context_size, gpu_layers=gpu_layers)
    failure = None
    try:
        yield provider
    except BaseException as exc:
        failure = exc
        raise
    finally:
        # Compatibility with existing structural providers that predate close().
        close = getattr(provider, 'close', None)
        if close is not None:
            try:
                close()
            except Exception:
                if failure is not None:
                    failure.add_note('Embedding provider cleanup failed; resources may remain open.')
                else:
                    raise RuntimeError('Operation completed, but embedding provider cleanup failed') from None


def embed_file(*, path: str, chunk_size: int, stride: int, model_path: str = DEFAULT_MODEL,
               context_size: int = 2048, gpu_layers: int = 0, batch_size: int = 1,
               provider_factory: Callable | None = None) -> dict:
    # Reject empty/invalid files before loading a potentially large local model.
    document = prepare_text_file(path, chunk_size, stride, batch_size)
    with open_embedding_provider(model_path=model_path, context_size=context_size,
                                 gpu_layers=gpu_layers, provider_factory=provider_factory) as provider:
        return embed_chunks(document, provider, batch_size)


def search_documents(*, query: str, limit: int = 10, model_path: str = DEFAULT_MODEL,
                     context_size: int = 2048, gpu_layers: int = 0) -> list[SearchResult]:
    with open_document_repository() as repository:
        with open_embedding_provider(model_path=model_path, context_size=context_size,
                                     gpu_layers=gpu_layers) as provider:
            return SearchDocuments(query=query, limit=limit, embedding_provider=provider,
                                   repository=repository).execute()


def ingest_file(*, path: str, chunk_size: int, stride: int, model_path: str = DEFAULT_MODEL,
                context_size: int = 2048, gpu_layers: int = 0, batch_size: int = 1) -> list[Document]:
    with create_context() as context:
        with open_embedding_provider(model_path=model_path, context_size=context_size,
                                     gpu_layers=gpu_layers) as provider:
            return IngestDocument(path=path, chunk_size=chunk_size, stride=stride,
                                  batch_size=batch_size, embedding_provider=provider,
                                  storage=context.storage, repository=context.documents).execute()
