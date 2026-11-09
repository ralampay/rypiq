from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import UUID, uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Document:
    content: str
    embedding: list[float]

    id: UUID = field(default_factory=uuid4)

    # Object storage reference
    storage_key: str | None = None

    # Original file information
    filename: str = ""
    content_type: str = ""
    source: str = ""
    source_id: UUID | None = None

    # RAG processing information
    chunk_index: int = 0
    embedding_model: str = ""
    metadata: dict = field(default_factory=dict)

    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
