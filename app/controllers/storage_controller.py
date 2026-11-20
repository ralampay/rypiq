from app.dependencies.config import ConfigDependency
from fastapi import APIRouter, File, Form, Request, UploadFile
from app.dependencies.uploads import staged_upload
from app.operations.storage.save import save_file
from app.operations.storage.list import list_files
from app.storage.factory import get_storage
from app.schemas.api import StorageOut

router = APIRouter(prefix='/storage', tags=['storage'])


@router.get('', response_model=list[StorageOut])
def index(config: ConfigDependency, prefix: str = ''):
    return list_files(storage=get_storage(config), prefix=prefix)


@router.post('', response_model=StorageOut, status_code=201)
def create(config: ConfigDependency, request: Request, file: UploadFile = File(), prefix: str = Form(default='documents')):
    with staged_upload(file, limit=config.require_api().max_upload_bytes) as path:
        return save_file(path=str(path), storage=get_storage(config), prefix=prefix)
