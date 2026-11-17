"""Configured Python entrypoints. Own resources here; keep commands injectable."""
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from app.database.factory import get_database
from app.models.application import Application
from app.operations.application import CreateApplication, get_application, list_applications, validate_local_application

from app.embeddings.base import EmbeddingProvider
from app.embeddings.config import DEFAULT_MODEL
from app.embeddings.factory import get_embedding_provider
from app.models.document import Document
from app.models.search import SearchResult
from app.operations.document import IngestDocument
from app.operations.embedding import embed_chunks, prepare_text_file
from app.operations.search import SearchDocuments


@contextmanager
def open_embedding_provider(*, model_path: str = DEFAULT_MODEL, context_size: int = 2048,
                            gpu_layers: int = 0, provider_factory: Callable | None = None
                            ) -> Iterator[EmbeddingProvider]:
    if type(context_size) is not int or context_size < 1:
        raise ValueError('context_size must be positive')
    factory = provider_factory if provider_factory is not None else get_embedding_provider
    provider = factory(model_path=model_path, context_size=context_size, gpu_layers=gpu_layers)
    failure = None
    try:
        yield provider
    except BaseException as exc:
        failure = exc
        raise
    finally:
        # Compatibility with existing structural providers that predate close().
        close = getattr(provider, 'close', None)
        if close is not None:
            try:
                close()
            except Exception:
                if failure is not None:
                    failure.add_note('Embedding provider cleanup failed; resources may remain open.')
                else:
                    raise RuntimeError('Operation completed, but embedding provider cleanup failed') from None


def embed_file(*, path: str, chunk_size: int, stride: int, model_path: str = DEFAULT_MODEL,
               context_size: int = 2048, gpu_layers: int = 0, batch_size: int = 1,
               provider_factory: Callable | None = None) -> dict:
    # Reject empty/invalid files before loading a potentially large local model.
    document = prepare_text_file(path, chunk_size, stride, batch_size)
    with open_embedding_provider(model_path=model_path, context_size=context_size,
                                 gpu_layers=gpu_layers, provider_factory=provider_factory) as provider:
        return embed_chunks(document, provider, batch_size)


def _application_model(application: Application, model_path: str | None) -> str:
    validate_local_application(application)
    if model_path is not None and Path(model_path).name != Path(application.embedding_model).name:
        raise ValueError('Model override must match the application embedding model filename')
    return model_path or application.embedding_model


def search_documents(*, query: str, application_id: UUID, limit: int = 10,
                     model_path: str | None = None,
                     context_size: int = 2048, gpu_layers: int = 0) -> list[SearchResult]:
    with get_database() as database:
        application = get_application(application_id=application_id, repository=database.applications())
        model = _application_model(application, model_path)
        with open_embedding_provider(model_path=model, context_size=context_size,
                                     gpu_layers=gpu_layers) as provider:
            return SearchDocuments(query=query, limit=limit, embedding_provider=provider,
                                   application=application, repository=database.documents()).execute()


def ingest_file(*, path: str, chunk_size: int, stride: int, application_id: UUID,
                model_path: str | None = None, context_size: int = 2048,
                gpu_layers: int = 0, batch_size: int = 1) -> list[Document]:
    from app.storage.factory import get_storage

    with get_database() as database:
        application = get_application(application_id=application_id, repository=database.applications())
        model = _application_model(application, model_path)
        with open_embedding_provider(model_path=model, context_size=context_size,
                                     gpu_layers=gpu_layers) as provider:
            return IngestDocument(path=path, chunk_size=chunk_size, stride=stride,
                                  batch_size=batch_size, embedding_provider=provider,
                                  application=application, storage=get_storage(),
                                  repository=database.documents()).execute()


def create_application(*, name: str, embedding_model: str, provision_type: str = 'local',
                       embedding_size: int | None = None, inference_model: str | None = None,
                       data: dict | None = None) -> Application:
    with get_database() as database:
        value = Application(name=name, embedding_model=embedding_model, provision_type=provision_type,
                            embedding_size=database.settings['dimensions'] if embedding_size is None else embedding_size,
                            inference_model=inference_model, data={} if data is None else data)
        return CreateApplication(application=value, repository=database.applications()).execute()


def show_application(*, application_id: UUID) -> Application:
    with get_database() as database:
        return get_application(application_id=application_id, repository=database.applications())


def enumerate_applications(*, limit: int = 100, offset: int = 0) -> list[Application]:
    with get_database() as database:
        return list_applications(repository=database.applications(), limit=limit, offset=offset)
