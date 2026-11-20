from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.models.document import DocumentStatus

PositiveInt = Annotated[int, Field(gt=0)]
Provider = Literal['local', 'openai', 'llama-server']


class RequestModel(BaseModel):
    model_config = ConfigDict(extra='forbid')


class ResponseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ApplicationCreate(RequestModel):
    name: str = Field(min_length=1)
    embedding_model: str = Field(min_length=1)
    embedding_size: int | None = Field(default=None, ge=1, le=16000)
    embedding_provision_type: Provider = 'local'
    inference_provision_type: Provider = 'local'
    inference_model: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)
    chunk_size: PositiveInt | None = None
    stride: PositiveInt | None = None


class ApplicationUpdate(RequestModel):
    name: str | None = Field(default=None, min_length=1)
    embedding_model: str | None = Field(default=None, min_length=1)
    embedding_size: int | None = Field(default=None, ge=1, le=16000)
    embedding_provision_type: Provider | None = None
    inference_provision_type: Provider | None = None
    inference_model: str | None = None
    data: dict[str, Any] | None = None
    chunk_size: PositiveInt | None = None
    stride: PositiveInt | None = None

    @model_validator(mode='after')
    def reject_nulls(self):
        if any(getattr(self, key) is None for key in self.model_fields_set - {'inference_model'}):
            raise ValueError('Only inference_model may be null')
        return self


class ApplicationOut(ResponseModel):
    id: UUID
    name: str
    embedding_model: str
    embedding_size: int
    embedding_provision_type: Provider
    inference_provision_type: Provider
    inference_model: str | None
    data: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ApplicationSummaryOut(ResponseModel):
    id: UUID
    name: str
    embedding_model: str
    embedding_size: int
    embedding_provision_type: Provider
    inference_provision_type: Provider
    inference_model: str | None
    document_count: int


class ChunkOut(ResponseModel):
    id: UUID
    chunk_index: int
    content: str
    metadata: dict[str, Any]


class DocumentOut(ResponseModel):
    id: UUID
    application_id: UUID
    name: str
    filename: str
    storage_key: str | None
    content_type: str
    metadata: dict[str, Any]
    status: DocumentStatus
    created_at: datetime
    updated_at: datetime
    embedding_count: int
    embeddings: list[ChunkOut]


class DocumentSummaryOut(ResponseModel):
    id: UUID
    name: str
    status: DocumentStatus
    embedding_count: int


class EmbeddingOut(ChunkOut):
    document_id: UUID
    embedding: list[float]
    model_name: str
    embedding_model: str
    embedding_size: int
    provision_type: Provider
    created_at: datetime
    updated_at: datetime


class ProcessRequest(RequestModel):
    batch_size: PositiveInt = 1


class JobOut(ResponseModel):
    id: UUID
    document_id: UUID
    batch_size: int
    state: Literal['queued', 'running', 'succeeded', 'failed']
    attempts: int
    created_at: datetime
    updated_at: datetime
    error: str | None


class JobAccepted(ResponseModel):
    job_id: UUID
    document_id: UUID
    state: str
    polling_url: str


class DocumentStatusOut(ResponseModel):
    document_id: UUID
    status: DocumentStatus
    job: JobOut | None


class Deleted(ResponseModel):
    deleted: bool


class SearchRequest(RequestModel):
    query: str = Field(min_length=1)
    limit: PositiveInt = 5
    metadata_filter: dict[str, Any] | None = None


class SearchOut(ResponseModel):
    embedding_id: UUID
    document_id: UUID
    distance: float
    content: str
    metadata: dict[str, Any]
    storage_key: str | None


class ChatRequest(RequestModel):
    prompt: str = Field(min_length=1)


class ChatSourceOut(ResponseModel):
    document_id: UUID
    name: str
    filename: str
    indicator: str


class ChatOut(ResponseModel):
    text: str
    model: str
    finish_reason: str | None
    usage: dict[str, int] | None
    sources: list[ChatSourceOut]


class ModelOut(ResponseModel):
    id: str
    provider: str
    capabilities: list[str]
    source: str
    availability: str
    model: str


class TextRequest(RequestModel):
    content: str


class TextOut(ResponseModel):
    medium: str
    content: str
    characters: int
    words: int


class TextChunkOut(ResponseModel):
    index: int
    start: int
    end: int
    text: str


class TextFileOut(ResponseModel):
    medium: str
    filename: str
    chunk_size: int
    stride: int
    total_characters: int
    chunk_count: int
    chunks: list[TextChunkOut]


class StorageOut(ResponseModel):
    key: str
    size: int
    content_type: str
    checksum: str


class HealthOut(ResponseModel):
    status: str


class CheckOut(ResponseModel):
    name: str
    status: str
    detail: str


class DoctorOut(ResponseModel):
    environment: str
    checks: list[CheckOut]
    incomplete_count: int
