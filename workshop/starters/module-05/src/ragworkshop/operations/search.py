from ragworkshop.context import ApplicationContext
from ragworkshop.database.models import SearchResult
from ragworkshop.embeddings.base import validate_vectors
from ragworkshop.processors.chunks import Metadata

def semantic_search(query: str, context: ApplicationContext, limit: int=5, filters: Metadata | None=None) -> list[SearchResult]:
    raise NotImplementedError('TODO: semantic_search')
