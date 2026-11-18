"""An original document; searchable text belongs to its embeddings."""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.embedding import Embedding


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Document:
    name: str
    application_id: UUID | None = None
    id: UUID = field(default_factory=uuid4)
    storage_key: str | None = None
    filename: str = ''
    content_type: str = ''
    source: str = ''
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


@dataclass
class DocumentDetails:
    document: Document
    embeddings: list['Embedding']


@dataclass(frozen=True)
class DocumentSummary:
    id: UUID
    name: str
    embedding_count: int
