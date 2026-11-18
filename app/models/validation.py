"""Validation shared by persistence boundaries and domain services."""
import json
from uuid import UUID
from typing import TYPE_CHECKING
from app.exceptions import InvalidIdentifierError, ValidationError

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.embedding import Embedding


def validate_metadata(value: dict) -> None:
    if not isinstance(value, dict):
        raise ValidationError('Metadata must be a JSON object')
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        raise ValidationError('Metadata must be a JSON object') from None


def normalize_document_name(name: str) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValidationError('Document name must be a nonempty string')
    return name.strip()


def validate_document(document: 'Document') -> None:
    document.name = normalize_document_name(document.name)
    if not isinstance(document.id, UUID) or not isinstance(document.application_id, UUID):
        raise InvalidIdentifierError('Document requires UUID identity and application ID')
    for name in ('filename', 'content_type', 'source'):
        if not isinstance(getattr(document, name), str):
            raise ValidationError(f'Document {name} must be a string')
    if document.storage_key is not None and not isinstance(document.storage_key, str):
        raise ValidationError('Document storage key must be a string or None')
    validate_metadata(document.metadata)


def validate_chunk(embedding: 'Embedding') -> list[float]:
    from app.models.embedding import validate_embedding
    if not isinstance(embedding.id, UUID) or not isinstance(embedding.document_id, UUID):
        raise InvalidIdentifierError('Embedding requires UUID identity and document ID')
    if not isinstance(embedding.content, str):
        raise ValidationError('Embedding content must be a string')
    if type(embedding.chunk_index) is not int or embedding.chunk_index < 0:
        raise ValidationError('Chunk index must be a nonnegative integer')
    validate_metadata(embedding.metadata)
    return validate_embedding(embedding.embedding, embedding.embedding_size)


def validate_id(value: UUID) -> None:
    if not isinstance(value, UUID):
        raise InvalidIdentifierError('ID must be a UUID')


def validate_limit(limit: int) -> None:
    if type(limit) is not int or limit < 1:
        raise ValidationError('Limit must be a positive integer')


def validate_pagination(limit: int, offset: int) -> None:
    validate_limit(limit)
    if type(offset) is not int or offset < 0:
        raise ValidationError('Offset must be a nonnegative integer')
