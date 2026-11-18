"""Configured Python entrypoints. Own resources here; keep commands injectable."""
from collections.abc import Callable, Iterator
from dataclasses import replace
from contextlib import contextmanager, AbstractContextManager
from pathlib import Path
from uuid import UUID

from app.database.factory import get_database
from app.models.application import Application, validate_application, chunking_settings, with_chunking_options
from app.services.application import ApplicationService
from app.providers.catalog import load_catalog, resolve_application, resolve_model, catalog_models
from app.providers.config import ProviderSettings, settings_for, validate_options, model_identity
from app.providers.contracts import ChatMessage, ChatResult, ModelInfo, InferenceProvider
from app.providers.factory import open_provider
from app.providers.session import LocalModelSession
from app.providers.discovery import merge_models
from app.operations.chat import Chat
from app.providers.normalization import validate_chat
from app.operations.application import (
    CreateApplication, get_application, list_applications, list_all_applications,
    delete_application as delete_application_record,
)

from app.embeddings.base import EmbeddingProvider
from app.embeddings.config import DEFAULT_MODEL
from app.embeddings.factory import get_embedding_provider
from app.models.document import Document
from app.models.search import SearchResult
from app.operations.document import IngestDocument
from app.operations.embedding import embed_chunks, prepare_text_file, get_embedding, delete_embedding as delete_embedding_record
from app.operations.search import SearchDocuments


def _catalog_for(*models: str | None) -> dict:
    return load_catalog() if any(isinstance(model, str) and model.startswith('@') for model in models) else {}


@contextmanager
def open_embedding_provider(*, model_path: str = DEFAULT_MODEL, context_size: int = 2048,
                            gpu_layers: int = 0, provider_factory: Callable | None = None,
                            model_session: LocalModelSession | None = None
                            ) -> Iterator[EmbeddingProvider]:
    if type(context_size) is not int or context_size < 1:
        raise ValueError('context_size must be positive')
    if model_session is not None:
        if provider_factory is not None:
            raise ValueError('Inject the provider factory into the model session instead')
        settings = ProviderSettings(provider='local', capability='embedding', model=model_path,
                                    embedding_size=768, context_size=context_size, gpu_layers=gpu_layers)
        with model_session.open_provider(settings) as provider:
            yield provider
        return
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
               context_size: int | None = None, gpu_layers: int | None = None, batch_size: int = 1,
               provider_factory: Callable | None = None, document_prefix: str | None = None,
               model_session: LocalModelSession | None = None) -> dict:
    # Reject empty/invalid files before loading a potentially large local model.
    document = prepare_text_file(path, chunk_size, stride, batch_size)
    resolved, defaults = resolve_model(model_path, 'local', 'embedding', _catalog_for(model_path))
    prefix = defaults.get('document_prefix', 'title: none | text: ') if document_prefix is None else document_prefix
    context_size = defaults.get('context_size', 2048) if context_size is None else context_size
    gpu_layers = defaults.get('gpu_layers', 0) if gpu_layers is None else gpu_layers
    validate_options({'context_size': context_size, 'gpu_layers': gpu_layers, 'document_prefix': prefix}, 'embedding')
    with open_embedding_provider(model_path=resolved, context_size=context_size,
                                 gpu_layers=gpu_layers, provider_factory=provider_factory,
                                 model_session=model_session) as provider:
        return embed_chunks(document, provider, batch_size, prefix)


def _application_settings(application: Application, capability: str, model_path: str | None = None,
                          context_size: int | None = None, gpu_layers: int | None = None) -> ProviderSettings:
    validate_application(application)
    settings = settings_for(application, capability)
    if model_path is not None:
        if application.provision_type != 'local':
            raise ValueError('Model path overrides are only supported for local applications')
        resolved, _ = resolve_model(model_path, 'local', capability, _catalog_for(model_path))
        if model_identity(resolved, 'local') != model_identity(settings.model, 'local'):
            raise ValueError('Model override must match the application model filename')
        settings = replace(settings, model=resolved)
    if settings.provider != 'local' and (context_size is not None or gpu_layers is not None):
        raise ValueError('Local runtime overrides cannot be used with remote providers')
    if context_size is not None:
        if type(context_size) is not int or context_size < 1:
            raise ValueError('context_size must be positive')
        settings = replace(settings, context_size=context_size)
    if gpu_layers is not None:
        if type(gpu_layers) is not int or gpu_layers < -1:
            raise ValueError('gpu_layers must be an integer of at least -1')
        settings = replace(settings, gpu_layers=gpu_layers)
    return settings


def _open_configured_provider(settings: ProviderSettings, model_session: LocalModelSession | None,
                              **dependencies) -> AbstractContextManager[EmbeddingProvider | InferenceProvider]:
    if model_session is not None and settings.provider == 'local':
        return model_session.open_provider(settings)
    return open_provider(settings, **dependencies)


def search_documents(*, query: str, application_id: UUID, limit: int = 10,
                     model_path: str | None = None,
                     context_size: int | None = None, gpu_layers: int | None = None,
                     model_session: LocalModelSession | None = None) -> list[SearchResult]:
    with get_database() as database:
        application = get_application(application_id=application_id, repository=database.applications())
        settings = _application_settings(application, 'embedding', model_path, context_size, gpu_layers)
        with _open_configured_provider(settings, model_session, embedding_factory=get_embedding_provider) as provider:
            return SearchDocuments(query=query, limit=limit, embedding_provider=provider,
                                   application=application, repository=database.documents()).execute()


def ingest_file(*, path: str, chunk_size: int, stride: int, application_id: UUID,
                model_path: str | None = None, context_size: int | None = None,
                gpu_layers: int | None = None, batch_size: int = 1,
                model_session: LocalModelSession | None = None) -> list[Document]:
    from app.storage.factory import get_storage

    with get_database() as database:
        application = get_application(application_id=application_id, repository=database.applications())
        settings = _application_settings(application, 'embedding', model_path, context_size, gpu_layers)
        with _open_configured_provider(settings, model_session, embedding_factory=get_embedding_provider) as provider:
            return IngestDocument(path=path, chunk_size=chunk_size, stride=stride,
                                  batch_size=batch_size, embedding_provider=provider,
                                  application=application, storage=get_storage(),
                                  repository=database.documents()).execute()


def create_embeddings(*, path: str, application_id: UUID, batch_size: int = 1) -> list[Document]:
    from app.storage.factory import get_storage

    with get_database() as database:
        application = get_application(application_id=application_id, repository=database.applications())
        settings = _application_settings(application, 'embedding')
        chunk_size, stride = chunking_settings(application.data)
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        with _open_configured_provider(settings, None, embedding_factory=get_embedding_provider) as provider:
            return IngestDocument(path=path, chunk_size=chunk_size, stride=stride,
                                  batch_size=batch_size, embedding_provider=provider,
                                  application=application, storage=get_storage(),
                                  repository=database.documents()).execute()


def show_embedding(*, embedding_id: UUID) -> Document:
    with get_database() as database:
        return get_embedding(embedding_id=embedding_id, repository=database.documents())


def delete_embedding(*, embedding_id: UUID) -> bool:
    with get_database() as database:
        return delete_embedding_record(embedding_id=embedding_id, repository=database.documents())


def create_application(*, name: str, embedding_model: str, provision_type: str = 'local',
                       embedding_size: int | None = None, inference_model: str | None = None,
                       data: dict | None = None, chunk_size: int | None = None,
                       stride: int | None = None) -> Application:
    with get_database() as database:
        value = Application(name=name, embedding_model=embedding_model, provision_type=provision_type,
                            embedding_size=database.settings['dimensions'] if embedding_size is None else embedding_size,
                            inference_model=inference_model, data=with_chunking_options(
                                {} if data is None else data, chunk_size=chunk_size, stride=stride))
        value = resolve_application(value, _catalog_for(value.embedding_model, value.inference_model))
        if value.provision_type == 'local':
            value = replace(value, embedding_model=str(Path(value.embedding_model).expanduser().resolve()),
                            inference_model=(str(Path(value.inference_model).expanduser().resolve())
                                             if value.inference_model is not None else None))
        return CreateApplication(application=value, repository=database.applications()).execute()


def show_application(*, application_id: UUID) -> Application:
    with get_database() as database:
        return get_application(application_id=application_id, repository=database.applications())


def enumerate_applications(*, limit: int = 100, offset: int = 0) -> list[Application]:
    with get_database() as database:
        return list_applications(repository=database.applications(), limit=limit, offset=offset)


def enumerate_all_applications() -> list[Application]:
    with get_database() as database:
        return list_all_applications(repository=database.applications())


def delete_application(*, application_id: UUID) -> bool:
    with get_database() as database:
        return delete_application_record(application_id=application_id, repository=database.applications())


def update_application(*, application: Application, chunk_size: int | None = None,
                       stride: int | None = None) -> Application:
    application = replace(application, data=with_chunking_options(
        application.data, chunk_size=chunk_size, stride=stride))
    resolved = resolve_application(application, _catalog_for(application.embedding_model, application.inference_model))
    with get_database() as database:
        return ApplicationService(repository=database.applications()).update(resolved)


def chat(*, messages: list[ChatMessage], application_id: UUID | None = None,
         model_path: str | None = None, max_tokens: int = 512,
         temperature: float | None = None, model_session: LocalModelSession | None = None) -> ChatResult:
    validate_chat(messages, max_tokens, temperature)
    if (application_id is None) == (model_path is None):
        raise ValueError('Choose exactly one application ID or standalone local model')
    if application_id is not None:
        with get_database() as database:
            value = get_application(application_id=application_id, repository=database.applications())
        settings = _application_settings(value, 'inference')
    else:
        value = resolve_application(Application(name='standalone', embedding_model='unused',
                                                inference_model=model_path), _catalog_for(model_path))
        settings = _application_settings(value, 'inference')
    with _open_configured_provider(settings, model_session) as provider:
        return Chat(messages=messages, provider=provider, max_tokens=max_tokens,
                    temperature=temperature).execute()


def list_models(*, application_id: UUID | None = None, discover: bool = False) -> list[ModelInfo]:
    configured = catalog_models(load_catalog())
    if application_id is None:
        if discover:
            raise ValueError('Remote discovery requires an application ID')
        return merge_models(configured, [])
    with get_database() as database:
        value = get_application(application_id=application_id, repository=database.applications())
    configured = [model for model in configured if model.provider == value.provision_type]
    discovered = []
    if discover and value.provision_type != 'local':
        seen = set()
        for capability in ('embedding', 'inference'):
            if capability == 'inference' and value.inference_model is None:
                if 'inference' not in value.data.get('provider_config', {}):
                    continue
                value = replace(value, inference_model='discovery')
            settings = settings_for(value, capability)
            endpoint = (settings.base_url, settings.api_key_env)
            if endpoint in seen:
                continue
            seen.add(endpoint)
            with open_provider(settings) as provider:
                discovered.extend(provider.list_models())
    return merge_models(configured, discovered)
