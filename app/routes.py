from fastapi import FastAPI

from app.controllers.query_controller import router as query_router


def register_routes(app: FastAPI) -> None:
    app.include_router(query_router)
