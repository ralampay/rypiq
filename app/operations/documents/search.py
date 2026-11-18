"""Backend-independent semantic search operation."""

import math

from app.models.application import Application
from app.models.embedding import validate_embedding
from app.models.application import validate_application
from app.providers.config import model_identity, embedding_preprocessing

from app.database.repositories.base import DocumentRepository
from app.embeddings.base import EmbeddingProvider
from app.models.search import SearchResult


QUERY_PREFIX = "task: search result | query: "


def _search(
    query: str,
    embedding_provider: EmbeddingProvider,
    repository: DocumentRepository,
    application: Application,
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

    validate_application(application)
    if embedding_provider.model_name != model_identity(application.embedding_model, application.provision_type):
        raise ValueError('Embedding provider model must match the application')
    # The provider accepts a list of strings.
    vectors = embedding_provider.embed([
        embedding_preprocessing(application.data, application.provision_type, application.embedding_size)[1] + query.strip()
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

    validate_embedding(vector, application.embedding_size)
    return repository.search_by_vector(
        embedding=vector,
        application_id=application.id,
        limit=limit,
    )


class SearchDocuments:
    def __init__(self, *, query: str, embedding_provider: EmbeddingProvider,
                 repository: DocumentRepository, application: Application, limit: int = 10) -> None:
        self.application = application
        self.query = query
        self.embedding_provider = embedding_provider
        self.repository = repository
        self.limit = limit

    def execute(self) -> list[SearchResult]:
        return _search(self.query, self.embedding_provider, self.repository, self.application, self.limit)


def semantic_search(query: str, embedding_provider: EmbeddingProvider,
                    repository: DocumentRepository, limit: int = 10, *,
                    application: Application) -> list[SearchResult]:
    """Compatibility wrapper; dependencies are still owned by the caller."""
    return SearchDocuments(query=query, embedding_provider=embedding_provider,
                           repository=repository, application=application, limit=limit).execute()
