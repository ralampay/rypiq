"""Validation shared across local and remote inference adapters."""

from app.exceptions import ProviderResponseError
from app.models.embedding import validate_embedding
from app.providers.types.chat import ChatResult
from app.providers.validation import validate_chat, validate_texts  # Compatibility exports.


def embedding_response(response, count: int, dimensions: int) -> list[list[float]]:
    try:
        body = response if isinstance(response, dict) else response.model_dump()
        data = sorted(body['data'], key=lambda item: item['index'])
        if len(data) != count or any(type(item['index']) is not int or item['index'] != i
                                     for i, item in enumerate(data)):
            raise ValueError
        return [validate_embedding(item['embedding'], dimensions) for item in data]
    except (TypeError, ValueError, KeyError, AttributeError):
        raise ProviderResponseError('Invalid embedding response from provider') from None


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
        raise ProviderResponseError('Invalid chat response from provider') from None
