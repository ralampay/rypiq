"""FastAPI composition only; importing the Python library does not create a server."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import exceptions as errors
from app.dependencies.auth import ServiceBoundary
from app.config import AppConfig, load_config
from app.logging import configure_logging, get_logger, shutdown_logging
from app.routes import register_routes


def error_status(error: Exception) -> int:
    if isinstance(error, (errors.NotFoundError, errors.StorageNotFoundError)):
        return 404
    if isinstance(error, (errors.ConflictError, errors.StorageAlreadyExistsError, errors.DatabaseIntegrityError)):
        return 409
    if isinstance(error, (errors.ValidationError, errors.ProcessingError, UnicodeError)):
        return 422
    if isinstance(error, errors.ProviderError):
        return 502
    if isinstance(error, (errors.ConfigurationError, errors.DatabaseError, errors.StorageError)):
        return 503
    return 500


async def error_response(request: Request, error: Exception):
    code = error_status(error)
    messages = {404: 'Resource not found', 409: 'Operation conflicts with current state',
                422: 'Invalid input or document content', 502: 'Model provider request failed',
                503: 'Service dependency unavailable', 500: 'Operation failed'}
    detail = {'detail': messages[code]}
    document_id = getattr(error, '_rypiq_document_id', None)
    if document_id is not None:
        detail['document_id'] = str(document_id)
    get_logger('api').error('request_failed', status=code, error_type=type(error).__name__)
    return JSONResponse(detail, status_code=code)


async def validation_response(request: Request, error: RequestValidationError):
    # Pydantic's default response echoes input values, including arbitrary provider data.
    return JSONResponse({'detail': [{'loc': list(item['loc']), 'type': item['type'],
                                    'msg': 'Invalid request value'} for item in error.errors()]}, status_code=422)


def create_app(config: AppConfig | None = None) -> FastAPI:
    config = load_config() if config is None else config
    api = config.require_api()
    secret, limit = api.secret_key, api.max_upload_bytes

    @asynccontextmanager
    async def lifespan(app):
        configure_logging(settings=config.require_logging())
        try:
            yield
        finally:
            shutdown_logging()

    app = FastAPI(title='RyPiq API', version='1.0', lifespan=lifespan, redoc_url=None)
    app.state.config = config
    app.add_middleware(ServiceBoundary, secret=secret, max_upload_bytes=limit)
    # The last registered middleware runs first: browser preflights must be
    # answered before service authentication, and errors need CORS headers too.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api.cors_origins,
        allow_credentials=False,
        allow_methods=['GET', 'POST', 'PATCH', 'DELETE', 'OPTIONS'],
        allow_headers=['Content-Type', 'X-App-Secret-Key'],
        expose_headers=['Location'],
    )
    app.add_exception_handler(errors.RyPiqError, error_response)
    app.add_exception_handler(UnicodeError, error_response)
    app.add_exception_handler(Exception, error_response)
    app.add_exception_handler(RequestValidationError, validation_response)
    register_routes(app)
    return app
