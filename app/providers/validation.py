"""Pure request validation shared by operations and adapters."""
from dataclasses import asdict
import math
from app.exceptions import ValidationError
from app.providers.types.chat import ChatMessage


def validate_chat(messages: list[ChatMessage], max_tokens: int, temperature: float | None) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValidationError('Chat requires at least one message')
    if any(not isinstance(m, ChatMessage) or m.role not in ('system', 'user', 'assistant') or
           not isinstance(m.content, str) or not m.content.strip() for m in messages):
        raise ValidationError('Chat messages require a supported role and nonempty text')
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValidationError('max_tokens must be a positive integer')
    if temperature is not None and (isinstance(temperature, bool) or
            not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or
            not 0 <= temperature <= 2):
        raise ValidationError('temperature must be finite and between 0 and 2')
    return [asdict(message) for message in messages]


def validate_texts(texts: list[str]) -> None:
    if not isinstance(texts, list) or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValidationError('Embedding inputs must be a list of nonempty strings')


