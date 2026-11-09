from app.database.repositories.base import (
    DocumentRepository,
)
from app.embeddings.base import EmbeddingProvider
from app.models.search import SearchResult


def semantic_search(
    query: str,
    embedding_provider: EmbeddingProvider,
    repository: DocumentRepository,
    limit: int = 10,
) -> list[SearchResult]:

    if not query.strip():
        raise ValueError("Query cannot be empty")

    # The existing provider must expose an operation
    # that returns a single query embedding.
    query_vector = embedding_provider.embed(query)

    return repository.search_by_vector(
        embedding=query_vector,
        limit=limit,
    )
