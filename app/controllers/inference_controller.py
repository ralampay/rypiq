from app.dependencies.config import ConfigDependency
from uuid import UUID
from fastapi import APIRouter
from app import application
from app.providers.types.chat import ChatMessage
from app.schemas.api import SearchRequest, SearchOut, ChatRequest, ChatOut, ModelOut

router = APIRouter(tags=['inference'])


@router.post('/applications/{application_id}/search', response_model=list[SearchOut])
def search(config: ConfigDependency, application_id: UUID, payload: SearchRequest):
    matches = application.search_documents(application_id=application_id, query=payload.query,
                                           limit=payload.limit, metadata_filters=payload.metadata_filter, config=config)
    return [dict(embedding_id=m.embedding.id, document_id=m.document.id, distance=m.distance,
                 content=m.embedding.content, metadata=m.embedding.metadata, storage_key=m.document.storage_key)
            for m in matches]


@router.post('/applications/{application_id}/chat', response_model=ChatOut)
def chat(config: ConfigDependency, application_id: UUID, payload: ChatRequest):
    return application.chat(application_id=application_id, messages=[ChatMessage('user', payload.prompt)], config=config)


@router.get('/models', response_model=list[ModelOut])
def models(config: ConfigDependency, application_id: UUID | None = None, discover: bool = False):
    return application.list_models(application_id=application_id, discover=discover, config=config)
