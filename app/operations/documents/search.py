"""Backend-independent semantic search operation."""

import math

from app.logging import log_operation

from app.exceptions import (
    InvalidEmbeddingError,
    ModelMismatchError,
    ProviderResponseError,
    ValidationError,
)
from app.models.application import Application
from app.models.embedding import validate_embedding
from app.models.application import validate_application
from app.providers.config import model_identity, embedding_preprocessing

from app.database.repositories.embedding import EmbeddingRepository
from app.providers.contracts.embedding import EmbeddingProvider
from app.models.search import SearchResult


QUERY_PREFIX = "task: search result | query: "


def _search(
    query: str,
    embedding_provider: EmbeddingProvider,
    repository: EmbeddingRepository,
    application: Application,
    limit: int = 10,
    metadata_filters: dict | None = None,
) -> list[SearchResult]:

    if not isinstance(query, str) or not query.strip():
        raise ValidationError("Query cannot be empty")

    if (
        not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit < 1
    ):
        raise ValidationError(
            "Limit must be a positive integer"
        )

    if metadata_filters is not None:
        from app.models.validation import validate_metadata
        validate_metadata(metadata_filters)
    validate_application(application)
    if embedding_provider.model_name != model_identity(application.embedding_model, application.provision_type):
        raise ModelMismatchError('Embedding provider model must match the application')
    # The provider accepts a list of strings.
    vectors = embedding_provider.embed([
        embedding_preprocessing(application.data, application.provision_type, application.embedding_size)[1] + query.strip()
    ])

    if (not isinstance(vectors, (list, tuple)) or len(vectors) != 1
            or not isinstance(vectors[0], (list, tuple)) or not vectors[0]):
        raise ProviderResponseError(
            "Expected exactly one nonempty query embedding"
        )

    vector = vectors[0]

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in vector
    ):
        raise ProviderResponseError(
            "Query embedding contains invalid values"
        )

    try:
        validate_embedding(vector, application.embedding_size)
    except InvalidEmbeddingError as error:
        raise ProviderResponseError(str(error)) from None
    return repository.search_by_vector(
        embedding=vector,
        application_id=application.id,
        limit=limit,
        metadata_filters=metadata_filters,
    )


class SearchDocuments:
    def __init__(self, *, query: str, embedding_provider: EmbeddingProvider,
                 repository: EmbeddingRepository, application: Application, limit: int = 10,
                 metadata_filters: dict | None = None) -> None:
        self.metadata_filters = metadata_filters
        self.application = application
        self.query = query
        self.embedding_provider = embedding_provider
        self.repository = repository
        self.limit = limit

    def execute(self) -> list[SearchResult]:
        with log_operation('search') as outcome:
            results = _search(self.query, self.embedding_provider, self.repository, self.application, self.limit, self.metadata_filters)
            outcome['result_count'] = len(results)
            outcome['application_id'] = str(self.application.id)
            return results


def semantic_search(query: str, embedding_provider: EmbeddingProvider,
                    repository: EmbeddingRepository, limit: int = 10, *,
                    application: Application, metadata_filters: dict | None = None) -> list[SearchResult]:
    """Compatibility wrapper; dependencies are still owned by the caller."""
    return SearchDocuments(query=query, embedding_provider=embedding_provider,
                           repository=repository, application=application, limit=limit, metadata_filters=metadata_filters).execute()
