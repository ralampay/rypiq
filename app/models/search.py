from dataclasses import dataclass

from app.models.document import Document


@dataclass(frozen=True)
class SearchResult:
    document: Document
    distance: float
