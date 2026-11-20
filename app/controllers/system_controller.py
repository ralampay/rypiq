from app.dependencies.config import ConfigDependency
from fastapi import APIRouter, Response
from app.diagnostics import check_configuration
from app.schemas.api import DoctorOut, HealthOut

router = APIRouter(tags=['system'])


@router.get('/health', response_model=HealthOut)
def health():
    return {'status': 'ok'}


@router.get('/doctor', response_model=DoctorOut)
def doctor(config: ConfigDependency, response: Response):
    report = check_configuration(config=config)
    response.status_code = 503 if report.incomplete_count else 200
    return report
