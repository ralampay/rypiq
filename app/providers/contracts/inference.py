"""Text-only inference; callers borrow providers and owners close them.

Return ChatResult and raise provider exceptions without raw upstream diagnostics.
Cleanup is not universally idempotent.
"""
from typing import Protocol
from app.providers.types.chat import ChatMessage, ChatResult


class InferenceProvider(Protocol):
    def chat(self, messages: list[ChatMessage], *, max_tokens: int = 512,
             temperature: float | None = None) -> ChatResult: ...
    def close(self) -> None: ...


