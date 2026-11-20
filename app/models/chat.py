"""Application chat answers and retrieved document provenance."""
from dataclasses import dataclass, field
from uuid import UUID

from app.providers.types.chat import ChatResult


@dataclass(frozen=True)
class ChatSource:
    document_id: UUID
    name: str
    filename: str
    source: str
    indicator: str


@dataclass(frozen=True)
class RagChatResult(ChatResult):
    sources: list[ChatSource] = field(default_factory=list)
