"""Text-only chat values shared by callers and adapters."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ChatMessage:
    role: Literal['system', 'user', 'assistant']
    content: str


@dataclass(frozen=True)
class ChatResult:
    text: str
    model: str
    finish_reason: str | None
    usage: dict[str, int] | None = None


