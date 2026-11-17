"""Persisted application configuration, independent of resource composition."""
from dataclasses import dataclass, field
from datetime import datetime
from secrets import token_urlsafe
from uuid import UUID, uuid4

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
            raise ValueError('Application name and embedding model must be nonempty')
    if application.provision_type not in ('local', 'openai', 'llama-server'):
        raise ValueError('Unsupported provision type')
    if type(application.embedding_size) is not int or not 1 <= application.embedding_size <= 16000:
        raise ValueError('Embedding size must be an integer between 1 and 16000')
    if application.inference_model is not None and not isinstance(application.inference_model, str):
        raise ValueError('Inference model must be a string or None')
    if not isinstance(application.data, dict):
        raise ValueError('Application data must be a JSON object')
    try:
        json.dumps(application.data, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError('Application data must be a JSON object') from None
