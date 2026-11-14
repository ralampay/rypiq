"""FastAPI application factory."""

from contextlib import asynccontextmanager
from threading import Lock

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.context import create_context
from app.embeddings.factory import get_embedding_provider
from app.environment import load_environment
from app.settings import Settings


def create_app() -> FastAPI:
    load_environment()

    @asynccontextmanager
    async def lifespan(app):
        settings = Settings.from_environment()
        context = create_context()
        provider = get_embedding_provider(model_path=settings.embedding_model)
        app.state.settings = settings
        app.state.context = context
        app.state.embedding_provider = provider
        # Serialize local inference to bound simultaneous model allocations.
        app.state.inference_lock = Lock()
        try:
            yield
        finally:
            provider.close()

    app = FastAPI(title="Rypiq API", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["POST"],
        allow_headers=["Content-Type"],
    )
    from app.routes import register_routes

    register_routes(app)
    return app
