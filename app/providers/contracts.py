"""Small public contracts and normalized provider values."""
from dataclasses import dataclass
from typing import Literal, Protocol

from app.embeddings.base import EmbeddingProvider


class ProviderError(RuntimeError):
    """Safe provider failure; upstream diagnostics are never included."""


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


@dataclass(frozen=True)
class ModelInfo:
    id: str
    provider: str
    capabilities: tuple[str, ...]
    source: str
    availability: str
    model: str


class InferenceProvider(Protocol):
    def chat(self, messages: list[ChatMessage], *, max_tokens: int = 512,
             temperature: float | None = None) -> ChatResult: ...
    def close(self) -> None: ...


class ModelDiscovery(Protocol):
    def list_models(self) -> list[ModelInfo]: ...
    def close(self) -> None: ...
