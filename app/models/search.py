from dataclasses import dataclass
from app.models.document import Document
from app.models.embedding import Embedding


@dataclass(frozen=True)
class SearchResult:
    embedding: Embedding
    document: Document
    distance: float
