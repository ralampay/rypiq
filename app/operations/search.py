"""Backend-independent semantic search operation."""

import math

from app.database.repositories.base import DocumentRepository
from app.embeddings.base import EmbeddingProvider
from app.models.search import SearchResult


QUERY_PREFIX = "task: search result | query: "


def semantic_search(
    query: str,
    embedding_provider: EmbeddingProvider,
    repository: DocumentRepository,
    limit: int = 10,
) -> list[SearchResult]:

    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty")

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit < 1
    ):
        raise ValueError(
            "Limit must be a positive integer"
        )

    # The provider accepts a list of strings.
    vectors = embedding_provider.embed([
        QUERY_PREFIX + query.strip()
    ])

    if len(vectors) != 1 or not vectors[0]:
        raise ValueError(
            "Expected exactly one nonempty query embedding"
        )

    vector = vectors[0]

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in vector
    ):
        raise ValueError(
            "Query embedding contains invalid values"
        )

    return repository.search_by_vector(
        embedding=vector,
        limit=limit,
    )
