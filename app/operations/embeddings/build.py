"""Build validated embedding records without persistence or provider effects."""
from uuid import UUID

from app.exceptions import EmptyDocumentError, ModelMismatchError
from app.models.application import Application
from app.models.embedding import Embedding, validate_embedding
from app.models.validation import validate_chunk
from app.providers.config import model_identity


def build_embeddings(document_id: UUID, result: dict, application: Application,
                     chunk_size: int, stride: int) -> list[Embedding]:
    if result['model'] != model_identity(application.embedding_model, application.embedding_provision_type):
        raise ModelMismatchError('Embedding provider model must match the application')
    if not result['chunks']:
        raise EmptyDocumentError('File has no nonempty text chunks')
    chunks = [Embedding(
        document_id=document_id, content=chunk['text'],
        embedding=validate_embedding(chunk['embedding'], application.embedding_size),
        chunk_index=chunk['index'], model_name=result['model'],
        embedding_model=application.embedding_model, embedding_size=application.embedding_size,
        provision_type=application.embedding_provision_type,
        metadata={'start': chunk['start'], 'end': chunk['end'],
                  'chunk_size': chunk_size, 'stride': stride},
    ) for chunk in result['chunks']]
    for chunk in chunks:
        validate_chunk(chunk)
    return chunks
