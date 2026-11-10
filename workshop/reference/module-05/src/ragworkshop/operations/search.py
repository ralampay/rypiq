from ragworkshop.context import ApplicationContext
from ragworkshop.database.models import SearchResult
from ragworkshop.embeddings.base import validate_vectors
from ragworkshop.processors.chunks import Metadata


def semantic_search(query: str, context: ApplicationContext, limit: int = 5,
                    filters: Metadata | None = None) -> list[SearchResult]:
    if not query.strip():
        raise ValueError("Query must not be empty")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    provider = context.embeddings
    context.documents.ensure_space(provider.model_name, provider.dimensions)
    vectors = provider.embed([query])
    validate_vectors(vectors, 1, provider.dimensions)
    return context.documents.search_by_vector(vectors[0], limit, filters)
