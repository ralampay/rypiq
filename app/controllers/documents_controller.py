from app.dependencies.config import ConfigDependency
from dataclasses import asdict
from uuid import UUID

from fastapi import APIRouter, Body, File, Form, Request, Response, UploadFile

from app import application
from app.dependencies.uploads import staged_upload
from app.models.document import DocumentStatus
from app.schemas.api import (DocumentOut, DocumentSummaryOut, DocumentStatusOut,
                             ProcessRequest, JobOut, JobAccepted, EmbeddingOut, Deleted)

router = APIRouter(tags=['documents'])


def document_response(value):
    return {**asdict(value.document), 'embedding_count': len(value.embeddings),
            'embeddings': sorted(value.embeddings, key=lambda item: (item.chunk_index, item.id))}


@router.get('/applications/{application_id}/documents', response_model=list[DocumentSummaryOut])
def index(config: ConfigDependency, application_id: UUID, status: DocumentStatus | None = None):
    return application.enumerate_documents(application_id=application_id, status=status, config=config)


@router.post('/applications/{application_id}/documents', response_model=DocumentOut, status_code=201)
def create(config: ConfigDependency, application_id: UUID, request: Request, name: str = Form(min_length=1), file: UploadFile = File()):
    with staged_upload(file, limit=config.require_api().max_upload_bytes) as path:
        return document_response(application.store_document(name=name, path=str(path), application_id=application_id, config=config))


@router.get('/documents/{document_id}', response_model=DocumentOut)
def show(config: ConfigDependency, document_id: UUID):
    return document_response(application.show_document(document_id=document_id, config=config))


@router.delete('/documents/{document_id}', response_model=Deleted)
def delete(config: ConfigDependency, document_id: UUID):
    return {'deleted': application.delete_document(document_id=document_id, config=config)}


@router.post('/documents/{document_id}/process', response_model=JobAccepted, status_code=202)
def process(config: ConfigDependency, document_id: UUID, response: Response, payload: ProcessRequest = Body(default=ProcessRequest())):
    job = application.enqueue_document(document_id=document_id, batch_size=payload.batch_size, config=config)
    url = f'/api/v1/processing-jobs/{job.id}'
    response.headers['Location'] = url
    return {'job_id': job.id, 'document_id': document_id, 'state': job.state, 'polling_url': url}


@router.get('/documents/{document_id}/status', response_model=DocumentStatusOut)
def status(config: ConfigDependency, document_id: UUID):
    document, job = application.document_processing_status(document_id=document_id, config=config)
    return {'document_id': document.id, 'status': document.status, 'job': job}


@router.get('/processing-jobs/{job_id}', response_model=JobOut)
def job(config: ConfigDependency, job_id: UUID):
    return application.show_processing_job(job_id=job_id, config=config)


@router.get('/embeddings/{embedding_id}', response_model=EmbeddingOut)
def embedding(config: ConfigDependency, embedding_id: UUID):
    return application.show_embedding(embedding_id=embedding_id, config=config)


@router.delete('/embeddings/{embedding_id}', response_model=Deleted)
def delete_embedding(config: ConfigDependency, embedding_id: UUID):
    return {'deleted': application.delete_embedding(embedding_id=embedding_id, config=config)}
