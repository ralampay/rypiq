"""Persisted application configuration, independent of resource composition."""
from dataclasses import dataclass, field
from datetime import datetime
from secrets import token_urlsafe
from uuid import UUID, uuid4

from app.exceptions import ValidationError
from app.models.document import utc_now


@dataclass
class Application:
    name: str
    embedding_model: str
    provision_type: str = 'local'
    embedding_size: int = 768
    inference_model: str | None = None
    data: dict = field(default_factory=dict)
    id: UUID = field(default_factory=uuid4)
    api_key: str = field(default_factory=lambda: token_urlsafe(32), repr=False)
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)


def validate_application(application: Application) -> None:
    import json

    for value in (application.name, application.embedding_model):
        if not isinstance(value, str) or not value.strip():
            raise ValidationError('Application name and embedding model must be nonempty')
    if application.provision_type not in ('local', 'openai', 'llama-server'):
        raise ValidationError('Unsupported provision type')
    if type(application.embedding_size) is not int or not 1 <= application.embedding_size <= 16000:
        raise ValidationError('Embedding size must be an integer between 1 and 16000')
    if application.inference_model is not None and not isinstance(application.inference_model, str):
        raise ValidationError('Inference model must be a string or None')
    if not isinstance(application.data, dict):
        raise ValidationError('Application data must be a JSON object')
    try:
        json.dumps(application.data, allow_nan=False)
    except (TypeError, ValueError):
        raise ValidationError('Application data must be a JSON object') from None

    chunking_settings(application.data)

    from app.providers.config import validate_provider_config
    validate_provider_config(application.data, application.embedding_size)


def chunking_settings(data: dict) -> tuple[int, int]:
    """Resolve saved chunking settings, including defaults for older applications."""
    chunking = data.get('chunking', {})
    if not isinstance(chunking, dict):
        raise ValidationError('Chunking settings must be a JSON object')
    size, stride = chunking.get('chunk_size', 500), chunking.get('stride', 400)
    if type(size) is not int or type(stride) is not int or not 0 < stride <= size:
        raise ValidationError('Chunk size and stride must be positive integers with stride <= chunk size')
    return size, stride


def with_chunking_options(data: dict, *, chunk_size: int | None = None,
                          stride: int | None = None) -> dict:
    """Merge explicit options without changing caller-owned application data."""
    if not isinstance(data, dict):
        raise ValidationError('Application data must be a JSON object')
    result = dict(data)
    if chunk_size is not None or stride is not None:
        chunking = result.get('chunking', {})
        if not isinstance(chunking, dict):
            raise ValidationError('Chunking settings must be a JSON object')
        chunking = dict(chunking)
        if chunk_size is not None:
            chunking['chunk_size'] = chunk_size
        if stride is not None:
            chunking['stride'] = stride
        result['chunking'] = chunking
    chunking_settings(result)
    return result
