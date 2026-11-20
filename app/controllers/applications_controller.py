from app.dependencies.config import ConfigDependency
from dataclasses import replace
from uuid import UUID

from fastapi import APIRouter

from app import application
from app.schemas.api import ApplicationCreate, ApplicationUpdate, ApplicationOut, ApplicationSummaryOut, Deleted

router = APIRouter(prefix='/applications', tags=['applications'])


@router.get('', response_model=list[ApplicationSummaryOut])
def index(config: ConfigDependency):
    return application.enumerate_application_summaries(config=config)


@router.post('', response_model=ApplicationOut, status_code=201)
def create(config: ConfigDependency, payload: ApplicationCreate):
    return application.create_application(**payload.model_dump(), config=config)


@router.get('/{application_id}', response_model=ApplicationOut)
def show(config: ConfigDependency, application_id: UUID):
    return application.show_application(application_id=application_id, config=config)


@router.patch('/{application_id}', response_model=ApplicationOut)
def update(config: ConfigDependency, application_id: UUID, payload: ApplicationUpdate):
    changes = payload.model_dump(exclude_unset=True)
    chunk_size, stride = changes.pop('chunk_size', None), changes.pop('stride', None)
    original = application.show_application(application_id=application_id, config=config)
    return application.update_application(application=replace(original, **changes),
                                          chunk_size=chunk_size, stride=stride, config=config)


@router.delete('/{application_id}', response_model=Deleted)
def delete(config: ConfigDependency, application_id: UUID):
    return {'deleted': application.delete_application(application_id=application_id, config=config)}
