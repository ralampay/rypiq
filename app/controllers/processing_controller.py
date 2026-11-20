from app.dependencies.config import ConfigDependency
from fastapi import APIRouter, File, Form, Request, UploadFile
from app.dependencies.uploads import staged_upload
from app.operations.processing.content import process_content
from app.operations.processing.file import process_text_file
from app.schemas.api import TextRequest, TextOut, TextFileOut

router = APIRouter(prefix='/processing', tags=['processing'])


@router.post('/text', response_model=TextOut)
def text(config: ConfigDependency, payload: TextRequest):
    return process_content(medium='text', content=payload.content)


@router.post('/text-file', response_model=TextFileOut)
def text_file(config: ConfigDependency, request: Request, file: UploadFile = File(), chunk_size: int = Form(gt=0), stride: int = Form(gt=0)):
    with staged_upload(file, limit=config.require_api().max_upload_bytes) as path:
        return {**process_text_file(path=str(path), chunk_size=chunk_size, stride=stride), 'filename': path.name}
