from fastapi import APIRouter, Depends, FastAPI
from app.dependencies.auth import require_app_secret
from app.controllers import (applications_controller, documents_controller, inference_controller,
                             processing_controller, storage_controller, system_controller)


def register_routes(app: FastAPI) -> None:
    api = APIRouter(prefix='/api/v1', dependencies=[Depends(require_app_secret)])
    for module in (applications_controller, documents_controller, inference_controller,
                   processing_controller, storage_controller, system_controller):
        api.include_router(module.router)
    app.include_router(api)
