"""Configured Python entrypoints. Own resources here; keep commands injectable."""
from collections.abc import Callable, Iterator
from dataclasses import replace
from contextlib import contextmanager, AbstractContextManager
from pathlib import Path
from uuid import UUID

from app.config import AppConfig, load_config
from app.logging import log_operation
from app.exceptions import (
    InvalidIdentifierError,
    ModelMismatchError,
    ProviderCleanupError,
    ValidationError,
)
from app.database.factory import get_database
from app.models.application import Application, ApplicationSummary, validate_application, with_chunking_options
from app.services.application import ApplicationService
from app.providers.catalog import load_catalog, resolve_model, catalog_models
from app.providers.config import ProviderSettings, validate_options, model_identity, embedding_preprocessing
from app.providers.types.chat import ChatMessage
from app.models.chat import RagChatResult
from app.providers.types.model import ModelInfo
from app.providers.contracts.inference import InferenceProvider
from app.providers.factory import open_capability_provider as open_provider, open_discovery_provider
from app.providers.resolution import settings_for, resolve_application, discovery_settings_for
from app.providers.session import LocalModelSession
from app.providers.discovery import merge_models
from app.operations.inference.chat import Chat, validate_rag_chat
from app.operations.applications.create import CreateApplication
from app.operations.applications.get import get_application
from app.operations.applications.list import list_applications, list_all_applications, list_application_summaries
from app.operations.applications.delete import delete_application as delete_application_record

from app.providers.contracts.embedding import EmbeddingProvider
from app.embeddings.config import DEFAULT_MODEL
from app.embeddings.factory import get_embedding_provider
from app.models.document import Document, DocumentDetails, DocumentSummary
from app.models.processing_job import ProcessingJob
from app.operations.documents.name import prepare_document_name
from app.operations.documents.list import list_documents
from app.models.embedding import Embedding
from app.operations.documents.get import get_document
from app.operations.documents.delete import delete_document as delete_document_record
from app.database.repositories.document import DocumentRepository
from app.storage.base import ObjectStorage
from app.models.search import SearchResult
from app.operations.documents.create import CreateDocument
from app.operations.documents.process import ProcessDocument
from app.operations.embeddings.embed_file import embed_chunks, prepare_text_file
from app.operations.embeddings.get import get_embedding
from app.operations.embeddings.delete import delete_embedding as delete_embedding_record
from app.operations.documents.search import SearchDocuments


def _catalog_for(*models: str | None, config: AppConfig | None = None) -> dict:
    return (load_catalog() if config is None else config.require_catalog()) if any(
        isinstance(model, str) and model.startswith('@') for model in models) else {}


@contextmanager
def open_embedding_provider(*, model_path: str = DEFAULT_MODEL, context_size: int = 2048,
                            gpu_layers: int = 0, provider_factory: Callable | None = None,
                            model_session: LocalModelSession | None = None
                            ) -> Iterator[EmbeddingProvider]:
    if type(context_size) is not int or context_size < 1:
        raise ValidationError('context_size must be positive')
    if model_session is not None:
        if provider_factory is not None:
            raise ValidationError('Inject the provider factory into the model session instead')
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
                    raise ProviderCleanupError('Operation completed, but embedding provider cleanup failed') from None


def embed_file(*, path: str, chunk_size: int, stride: int, model_path: str = DEFAULT_MODEL,
               context_size: int | None = None, gpu_layers: int | None = None, batch_size: int = 1,
               provider_factory: Callable | None = None, document_prefix: str | None = None,
               model_session: LocalModelSession | None = None, config: AppConfig | None = None) -> dict:
    config = load_config() if config is None else config
    with log_operation('embed_file'):
        # Reject empty/invalid files before loading a potentially large local model.
        document = prepare_text_file(path, chunk_size, stride, batch_size)
        resolved, defaults = resolve_model(model_path, 'local', 'embedding', _catalog_for(model_path, config=config))
        prefix = defaults.get('document_prefix', 'title: none | text: ') if document_prefix is None else document_prefix
        context_size = defaults.get('context_size', 2048) if context_size is None else context_size
        gpu_layers = defaults.get('gpu_layers', 0) if gpu_layers is None else gpu_layers
        validate_options({'context_size': context_size, 'gpu_layers': gpu_layers, 'document_prefix': prefix}, 'embedding')
        with open_embedding_provider(model_path=resolved, context_size=context_size,
                                     gpu_layers=gpu_layers, provider_factory=provider_factory,
                                     model_session=model_session) as provider:
            return embed_chunks(document, provider, batch_size, prefix)


def _application_settings(application: Application, capability: str, model_path: str | None = None,
                          context_size: int | None = None, gpu_layers: int | None = None, *, config: AppConfig | None = None) -> ProviderSettings:
    validate_application(application)
    settings = settings_for(application, capability)
    if model_path is not None:
        if settings.provider != 'local':
            raise ValidationError('Model path overrides are only supported for local applications')
        resolved, _ = resolve_model(model_path, 'local', capability, _catalog_for(model_path, config=config))
        if model_identity(resolved, 'local') != model_identity(settings.model, 'local'):
            raise ModelMismatchError('Model override must match the application model filename')
        settings = replace(settings, model=resolved)
    if settings.provider != 'local' and (context_size is not None or gpu_layers is not None):
        raise ValidationError('Local runtime overrides cannot be used with remote providers')
    if context_size is not None:
        if type(context_size) is not int or context_size < 1:
            raise ValidationError('context_size must be positive')
        settings = replace(settings, context_size=context_size)
    if gpu_layers is not None:
        if type(gpu_layers) is not int or gpu_layers < -1:
            raise ValidationError('gpu_layers must be an integer of at least -1')
        settings = replace(settings, gpu_layers=gpu_layers)
    return settings


def _open_configured_provider(settings: ProviderSettings, model_session: LocalModelSession | None,
                              config: AppConfig | None = None, **dependencies) -> AbstractContextManager[EmbeddingProvider | InferenceProvider]:
    if model_session is not None and settings.provider == 'local':
        return model_session.open_provider(settings)
    return open_provider(settings, **dependencies, **({"credentials": config.credentials} if config is not None else {}))


def search_documents(*, query: str, application_id: UUID, limit: int = 10,
                     model_path: str | None = None, metadata_filters: dict | None = None,
                     context_size: int | None = None, gpu_layers: int | None = None,
                     model_session: LocalModelSession | None = None, config: AppConfig | None = None) -> list[SearchResult]:
    config = load_config() if config is None else config
    with get_database(config) as database:
        application = get_application(application_id=application_id, repository=database.applications())
        settings = _application_settings(application, 'embedding', model_path, context_size, gpu_layers, config=config)
        with _open_configured_provider(settings, model_session, embedding_factory=get_embedding_provider, config=config) as provider:
            return SearchDocuments(query=query, limit=limit, embedding_provider=provider,
                                   application=application, repository=database.embeddings(),
                                   metadata_filters=metadata_filters).execute()


def ingest_file(*, name: str, path: str, chunk_size: int, stride: int, application_id: UUID,
                model_path: str | None = None, context_size: int | None = None,
                gpu_layers: int | None = None, batch_size: int = 1,
                model_session: LocalModelSession | None = None, config: AppConfig | None = None) -> DocumentDetails:
    config = load_config() if config is None else config
    from app.storage.factory import get_storage

    if type(batch_size) is not int or batch_size < 1:
        raise ValidationError('batch_size must be a positive integer')
    with get_database(config) as database:
        application = get_application(application_id=application_id, repository=database.applications())
        name = prepare_document_name(name=name, application_id=application.id, repository=database.documents())
        storage = get_storage(config)
        created = CreateDocument(name=name, path=path, application=application,
            repository=database.documents(), storage=storage).execute()
        return ProcessDocument(document_id=created.document.id, repository=database.documents(),
            applications=database.applications(), storage=storage, batch_size=batch_size,
            chunk_size=chunk_size, stride=stride,
            provider_scope=lambda current: _open_configured_provider(
                _application_settings(current, 'embedding', model_path, context_size, gpu_layers, config=config),
                model_session, embedding_factory=get_embedding_provider, config=config)).execute()


def store_document(*, name: str, path: str, application_id: UUID, config: AppConfig | None = None) -> DocumentDetails:
    """Persist an original without starting extraction or embedding."""
    config = load_config() if config is None else config
    from app.storage.factory import get_storage
    with get_database(config) as database:
        value = get_application(application_id=application_id, repository=database.applications())
        return CreateDocument(name=name, path=path, application=value,
                              repository=database.documents(), storage=get_storage(config), source_name=Path(path).name).execute()


def enqueue_document(*, document_id: UUID, batch_size: int = 1, config: AppConfig | None = None) -> ProcessingJob:
    config = load_config() if config is None else config
    with get_database(config) as database:
        return database.processing_jobs().enqueue(document_id=document_id, batch_size=batch_size)


def show_processing_job(*, job_id: UUID, config: AppConfig | None = None) -> ProcessingJob:
    config = load_config() if config is None else config
    from app.exceptions import NotFoundError
    with get_database(config) as database:
        job = database.processing_jobs().get(job_id)
        if job is None:
            raise NotFoundError(job_id, resource_id=job_id)
        return job


def document_processing_status(*, document_id: UUID, config: AppConfig | None = None) -> tuple[Document, ProcessingJob | None]:
    config = load_config() if config is None else config
    from app.exceptions import DocumentNotFoundError
    with get_database(config) as database:
        document = database.documents().get(document_id)
        if document is None:
            raise DocumentNotFoundError(document_id, resource_id=document_id)
        return document, database.processing_jobs().latest(document_id)


def create_document(*, name: str, path: str, application_id: UUID, batch_size: int = 1, config: AppConfig | None = None) -> DocumentDetails:
    config = load_config() if config is None else config
    from app.storage.factory import get_storage

    if type(batch_size) is not int or batch_size < 1:
        raise ValidationError('batch_size must be a positive integer')
    with get_database(config) as database:
        application = get_application(application_id=application_id, repository=database.applications())
        name = prepare_document_name(name=name, application_id=application.id, repository=database.documents())
        storage = get_storage(config)
        created = CreateDocument(name=name, path=path, application=application,
            repository=database.documents(), storage=storage).execute()
        return ProcessDocument(document_id=created.document.id, repository=database.documents(),
            applications=database.applications(), storage=storage, batch_size=batch_size,
            provider_scope=lambda current: _open_configured_provider(
                _application_settings(current, 'embedding', config=config), None,
                embedding_factory=get_embedding_provider, config=config)).execute()


def process_document(*, document_id: UUID, batch_size: int = 1,
                     model_session: LocalModelSession | None = None, config: AppConfig | None = None) -> DocumentDetails:
    config = load_config() if config is None else config
    from app.storage.factory import get_storage

    with get_database(config) as database:
        return ProcessDocument(document_id=document_id, repository=database.documents(),
            applications=database.applications(), storage=get_storage(config), batch_size=batch_size,
            provider_scope=lambda current: _open_configured_provider(
                _application_settings(current, 'embedding', config=config), model_session,
                embedding_factory=get_embedding_provider, config=config)).execute()


def enumerate_documents(*, application_id: UUID, status: str | None = None, config: AppConfig | None = None) -> list[DocumentSummary]:
    config = load_config() if config is None else config
    with get_database(config) as database:
        return list_documents(application_id=application_id, repository=database.documents(),
                              applications=database.applications(),
                              **({'status': status} if status is not None else {}))


def show_document(*, document_id: UUID, config: AppConfig | None = None) -> DocumentDetails:
    config = load_config() if config is None else config
    with get_database(config) as database:
        return get_document(document_id=document_id, repository=database.documents())


def delete_document(*, document_id: UUID, config: AppConfig | None = None) -> bool:
    config = load_config() if config is None else config
    with get_database(config) as database:
        return delete_document_record(document_id=document_id, repository=database.documents())


def show_embedding(*, embedding_id: UUID, config: AppConfig | None = None) -> Embedding:
    config = load_config() if config is None else config
    with get_database(config) as database:
        return get_embedding(embedding_id=embedding_id, repository=database.embeddings())


def delete_embedding(*, embedding_id: UUID, config: AppConfig | None = None) -> bool:
    config = load_config() if config is None else config
    with get_database(config) as database:
        return delete_embedding_record(embedding_id=embedding_id, repository=database.embeddings())


def _normalize_local_models(value: Application) -> Application:
    paths = {}
    for capability in ('embedding', 'inference'):
        model = getattr(value, capability + '_model')
        if model is not None and getattr(value, capability + '_provision_type') == 'local':
            paths[capability + '_model'] = str(Path(model).expanduser().resolve())
    return replace(value, **paths)


def create_application(*, name: str, embedding_model: str, embedding_provision_type: str = 'local',
                       inference_provision_type: str = 'local',
                       embedding_size: int | None = None, inference_model: str | None = None,
                       data: dict | None = None, chunk_size: int | None = None,
                       stride: int | None = None, config: AppConfig | None = None) -> Application:
    config = load_config() if config is None else config
    with get_database(config) as database:
        value = Application(name=name, embedding_model=embedding_model,
                            embedding_provision_type=embedding_provision_type,
                            inference_provision_type=inference_provision_type,
                            embedding_size=database.settings['dimensions'] if embedding_size is None else embedding_size,
                            inference_model=inference_model, data=with_chunking_options(
                                {} if data is None else data, chunk_size=chunk_size, stride=stride))
        value = resolve_application(value, _catalog_for(value.embedding_model, value.inference_model, config=config))
        value = _normalize_local_models(value)
        return CreateApplication(application=value, repository=database.applications()).execute()


def show_application(*, application_id: UUID, config: AppConfig | None = None) -> Application:
    config = load_config() if config is None else config
    with get_database(config) as database:
        return get_application(application_id=application_id, repository=database.applications())


def enumerate_applications(*, limit: int = 100, offset: int = 0, config: AppConfig | None = None) -> list[Application]:
    config = load_config() if config is None else config
    with get_database(config) as database:
        return list_applications(repository=database.applications(), limit=limit, offset=offset)


def enumerate_all_applications(*, config: AppConfig | None = None) -> list[Application]:
    config = load_config() if config is None else config
    with get_database(config) as database:
        return list_all_applications(repository=database.applications())


def enumerate_application_summaries(*, config: AppConfig | None = None) -> list[ApplicationSummary]:
    config = load_config() if config is None else config
    with get_database(config) as database:
        return list_application_summaries(repository=database.applications())


def delete_application(*, application_id: UUID, config: AppConfig | None = None) -> bool:
    config = load_config() if config is None else config
    with get_database(config) as database:
        return delete_application_record(application_id=application_id, repository=database.applications())


def update_application(*, application: Application, chunk_size: int | None = None,
                       stride: int | None = None, config: AppConfig | None = None) -> Application:
    config = load_config() if config is None else config
    application = replace(application, data=with_chunking_options(
        application.data, chunk_size=chunk_size, stride=stride))
    resolved = resolve_application(application, _catalog_for(application.embedding_model, application.inference_model, config=config))
    with get_database(config) as database:
        service = ApplicationService(repository=database.applications())
        original = service.get(resolved.id)
        paths = {}
        for capability in ('embedding', 'inference'):
            field = capability + '_model'
            provider_field = capability + '_provision_type'
            model = getattr(resolved, field)
            if (model is not None and getattr(resolved, provider_field) == 'local'
                    and (model != getattr(original, field)
                         or getattr(resolved, provider_field) != getattr(original, provider_field))):
                paths[field] = str(Path(model).expanduser().resolve())
        return service.update(replace(resolved, **paths))


def chat(*, messages: list[ChatMessage], application_id: UUID, max_tokens: int = 512,
         temperature: float | None = None, model_session: LocalModelSession | None = None, config: AppConfig | None = None) -> RagChatResult:
    """Retrieve application documents and generate a grounded answer; own per-call resources."""
    config = load_config() if config is None else config
    validate_rag_chat(messages, max_tokens, temperature)
    with get_database(config) as database:
        value = get_application(application_id=application_id, repository=database.applications())
        inference_settings = _application_settings(value, 'inference', config=config)
        embedding_settings = _application_settings(value, 'embedding', config=config)
        with _open_configured_provider(embedding_settings, model_session, config=config) as embedding_provider:
            with _open_configured_provider(inference_settings, model_session, config=config) as provider:
                return Chat(messages=messages, application=value, repository=database.embeddings(),
                            embedding_provider=embedding_provider, provider=provider,
                            max_tokens=max_tokens, temperature=temperature).execute()


def list_models(*, application_id: UUID | None = None, discover: bool = False, config: AppConfig | None = None) -> list[ModelInfo]:
    config = load_config() if config is None else config
    configured = catalog_models(config.require_catalog())
    if application_id is None:
        if discover:
            raise InvalidIdentifierError('Remote discovery requires an application ID')
        return merge_models(configured, [])
    with get_database(config) as database:
        value = get_application(application_id=application_id, repository=database.applications())
    providers = {value.embedding_provision_type, value.inference_provision_type}
    configured = [model for model in configured if model.provider in providers]
    discovered = []
    if discover:
        seen = set()
        for capability in ('embedding', 'inference'):
            if getattr(value, capability + '_provision_type') == 'local':
                continue
            if capability == 'inference' and value.inference_model is None:
                if 'inference' not in value.data.get('provider_config', {}):
                    continue
            settings = discovery_settings_for(value, capability)
            endpoint = (settings.provider, settings.base_url, settings.api_key_env)
            if endpoint in seen:
                continue
            seen.add(endpoint)
            with open_discovery_provider(settings, credentials=config.credentials) as provider:
                discovered.extend(provider.list_models())
    return merge_models(configured, discovered)


def embed_text_file(path: str, chunk_size: int, stride: int,
                    model_path: str = DEFAULT_MODEL, context_size: int | None = None,
                    gpu_layers: int | None = None, batch_size: int = 1,
                    provider_factory: Callable | None = None, document_prefix: str | None = None, *,
                    model_session: 'LocalModelSession | None' = None, config: AppConfig | None = None) -> dict:
    """Compatibility entrypoint; borrow session models or own per-call providers."""
    config = load_config() if config is None else config
    return embed_file(path=path, chunk_size=chunk_size, stride=stride,
                      model_path=model_path, context_size=context_size,
                      gpu_layers=gpu_layers, batch_size=batch_size,
                      provider_factory=provider_factory, document_prefix=document_prefix, model_session=model_session, config=config)


def ingest_document(path: str, storage: ObjectStorage, repository: DocumentRepository,
                    chunk_size: int, stride: int, *, name: str, application: Application,
                    model_session: 'LocalModelSession | None' = None, **embedding_options) -> DocumentDetails:
    """Local compatibility entrypoint using the durable creation/processing lifecycle."""
    validate_application(application)
    if application.embedding_provision_type != "local":
        raise ValidationError("Use IngestDocument with an injected provider for remote embeddings")
    batch_size = embedding_options.pop('batch_size', 1)
    if type(batch_size) is not int or batch_size < 1:
        raise ValidationError('batch_size must be a positive integer')
    prefix = embedding_options.pop('document_prefix', None)
    expected_prefix = embedding_preprocessing(application.data, application.embedding_provision_type, application.embedding_size)[0]
    if prefix is not None and prefix != expected_prefix:
        raise ValidationError('Document prefix must match the application configuration')
    @contextmanager
    def provider_scope(current):
        settings = settings_for(current, 'embedding')
        options = dict(model_path=settings.model, context_size=settings.context_size, gpu_layers=settings.gpu_layers)
        options.update(embedding_options)
        with open_embedding_provider(model_session=model_session, **options) as provider:
            yield provider
    created = CreateDocument(name=name, path=path, application=application,
        repository=repository, storage=storage).execute()
    return ProcessDocument(document_id=created.document.id, repository=repository,
        application=application, storage=storage, provider_scope=provider_scope,
        batch_size=batch_size, chunk_size=chunk_size, stride=stride).execute()
