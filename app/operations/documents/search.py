"""Backend-independent semantic search operation."""

import math

from app.database.repositories.embedding import EmbeddingRepository
from app.exceptions import (
    InvalidEmbeddingError,
    ModelMismatchError,
    ProviderResponseError,
    ValidationError,
)
from app.logging import log_operation
from app.models.application import Application, validate_application
from app.models.embedding import validate_embedding
from app.models.processing import embedding_settings
from app.models.search import SearchResult
from app.models.validation import validate_metadata
from app.providers.config import embedding_preprocessing, model_identity
from app.providers.contracts.embedding import EmbeddingProvider


QUERY_PREFIX = "task: search result | query: "


class SearchDocuments:
    def __init__(
        self,
        *,
        query: str,
        embedding_provider: EmbeddingProvider,
        repository: EmbeddingRepository,
        application: Application,
        limit: int = 10,
        metadata_filters: dict | None = None,
    ) -> None:
        self.metadata_filters = metadata_filters
        self.application = application
        self.query = query
        self.embedding_provider = embedding_provider
        self.repository = repository
        self.limit = limit

    def execute(self) -> list[SearchResult]:
        with log_operation('search') as outcome:
            self._validate_inputs()
            vector = self._embed_query()
            results = self.repository.search_by_vector(
                embedding=vector,
                application_id=self.application.id,
                limit=self.limit,
                metadata_filters=self.metadata_filters,
                expected_settings=embedding_settings(self.application),
            )
            outcome['result_count'] = len(results)
            outcome['application_id'] = str(self.application.id)
            return results

    def _validate_inputs(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValidationError("Query cannot be empty")

        if not isinstance(self.limit, int) or isinstance(self.limit, bool) or self.limit < 1:
            raise ValidationError("Limit must be a positive integer")

        if self.metadata_filters is not None:
            validate_metadata(self.metadata_filters)
        validate_application(self.application)
        if self.embedding_provider.model_name != model_identity(
            self.application.embedding_model, self.application.embedding_provision_type,
        ):
            raise ModelMismatchError('Embedding provider model must match the application')

    def _embed_query(self) -> list[float] | tuple[float, ...]:
        query_prefix = embedding_preprocessing(
            self.application.data,
            self.application.embedding_provision_type,
            self.application.embedding_size,
        )[1]
        vectors = self.embedding_provider.embed([query_prefix + self.query.strip()])

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
            validate_embedding(vector, self.application.embedding_size)
        except InvalidEmbeddingError as error:
            raise ProviderResponseError(str(error)) from None
        return vector


def semantic_search(query: str, embedding_provider: EmbeddingProvider,
                    repository: EmbeddingRepository, limit: int = 10, *,
                    application: Application, metadata_filters: dict | None = None) -> list[SearchResult]:
    """Compatibility wrapper; dependencies are still owned by the caller."""
    return SearchDocuments(
        query=query,
        embedding_provider=embedding_provider,
        repository=repository,
        application=application,
        limit=limit,
        metadata_filters=metadata_filters,
    ).execute()
