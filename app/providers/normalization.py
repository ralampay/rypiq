"""Validation shared across local and remote inference adapters."""
from dataclasses import asdict
import math

from app.models.embedding import validate_embedding
from app.providers.contracts import ChatMessage, ChatResult, ProviderError


def validate_chat(messages: list[ChatMessage], max_tokens: int, temperature: float | None) -> list[dict]:
    if not isinstance(messages, list) or not messages:
        raise ValueError('Chat requires at least one message')
    if any(not isinstance(m, ChatMessage) or m.role not in ('system', 'user', 'assistant') or
           not isinstance(m.content, str) or not m.content.strip() for m in messages):
        raise ValueError('Chat messages require a supported role and nonempty text')
    if type(max_tokens) is not int or max_tokens < 1:
        raise ValueError('max_tokens must be a positive integer')
    if temperature is not None and (isinstance(temperature, bool) or
            not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or
            not 0 <= temperature <= 2):
        raise ValueError('temperature must be finite and between 0 and 2')
    return [asdict(message) for message in messages]


def validate_texts(texts: list[str]) -> None:
    if not isinstance(texts, list) or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError('Embedding inputs must be a list of nonempty strings')


def embedding_response(response, count: int, dimensions: int) -> list[list[float]]:
    try:
        body = response if isinstance(response, dict) else response.model_dump()
        data = sorted(body['data'], key=lambda item: item['index'])
        if len(data) != count or any(type(item['index']) is not int or item['index'] != i
                                     for i, item in enumerate(data)):
            raise ValueError
        return [validate_embedding(item['embedding'], dimensions) for item in data]
    except (TypeError, ValueError, KeyError, AttributeError):
        raise ProviderError('Invalid embedding response from provider') from None


def chat_response(response, model: str) -> ChatResult:
    try:
        body = response if isinstance(response, dict) else response.model_dump()
        choice = body['choices'][0]
        text = choice['message']['content']
        reported = body.get('model') or model
        reason = choice.get('finish_reason')
        if not isinstance(text, str) or not isinstance(reported, str) or (reason is not None and not isinstance(reason, str)):
            raise ValueError
        usage = body.get('usage')
        if usage is not None:
            if not isinstance(usage, dict):
                raise ValueError
            usage = {key: usage[key] for key in ('prompt_tokens', 'completion_tokens', 'total_tokens') if key in usage}
            if any(type(value) is not int or value < 0 for value in usage.values()):
                raise ValueError
        return ChatResult(text, reported, reason, usage)
    except (TypeError, ValueError, KeyError, IndexError, AttributeError):
        raise ProviderError('Invalid chat response from provider') from None
