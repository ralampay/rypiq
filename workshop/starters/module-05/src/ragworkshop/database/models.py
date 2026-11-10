from dataclasses import dataclass
from ragworkshop.processors.chunks import Metadata


@dataclass(frozen=True)
class Document:
    id: str
    source_id: str
    content: str
    embedding: list[float]
    storage_key: str
    metadata: Metadata


@dataclass(frozen=True)
class SearchResult:
    document: Document
    distance: float
