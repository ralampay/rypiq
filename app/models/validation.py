"""Validation shared by persistence boundaries and domain services."""
import json
from uuid import UUID
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.document import Document
    from app.models.embedding import Embedding


def validate_metadata(value: dict) -> None:
    if not isinstance(value, dict):
        raise ValueError('Metadata must be a JSON object')
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError):
        raise ValueError('Metadata must be a JSON object') from None


def validate_document(document: 'Document') -> None:
    if not isinstance(document.id, UUID) or not isinstance(document.application_id, UUID):
        raise ValueError('Document requires UUID identity and application ID')
    for name in ('filename', 'content_type', 'source'):
        if not isinstance(getattr(document, name), str):
            raise ValueError(f'Document {name} must be a string')
    if document.storage_key is not None and not isinstance(document.storage_key, str):
        raise ValueError('Document storage key must be a string or None')
    validate_metadata(document.metadata)


def validate_chunk(embedding: 'Embedding') -> list[float]:
    from app.models.embedding import validate_embedding
    if not isinstance(embedding.id, UUID) or not isinstance(embedding.document_id, UUID):
        raise ValueError('Embedding requires UUID identity and document ID')
    if not isinstance(embedding.content, str):
        raise ValueError('Embedding content must be a string')
    if type(embedding.chunk_index) is not int or embedding.chunk_index < 0:
        raise ValueError('Chunk index must be a nonnegative integer')
    validate_metadata(embedding.metadata)
    return validate_embedding(embedding.embedding, embedding.embedding_size)
