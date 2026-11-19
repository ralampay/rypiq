"""CLI adapters: map parsed arguments to Python entrypoints and JSON shapes."""
from argparse import Namespace
from dataclasses import asdict, replace
from app.exceptions import ValidationError
from app.providers.types.chat import ChatMessage

from app import application
from app.models.document import DocumentDetails
from app.models.embedding import Embedding
from app.database import commands as database_commands
from app.operations.processing.content import process_content
from app.operations.storage.list import list_files
from app.operations.storage.save import save_file
from app.operations.processing.file import process_text_file
from app.storage.factory import get_storage
from app.storage.models import StoredObject


def model_options(args: Namespace) -> dict:
    return dict(model_path=args.model, context_size=args.context_size, gpu_layers=args.gpu_layers)


def process(args: Namespace) -> dict:
    if args.medium == 'text-file':
        if args.chunk_size is None or args.stride is None:
            raise ValidationError('text-file requires --chunk_size and --stride')
        return process_text_file(path=args.content, chunk_size=args.chunk_size, stride=args.stride)
    if args.chunk_size is not None or args.stride is not None:
        raise ValidationError('chunking options require text-file')
    return process_content(medium=args.medium, content=args.content)


def stored_object(value: StoredObject) -> dict:
    return dict(key=value.key, size=value.size, content_type=value.content_type, checksum=value.checksum)


def storage_save(args: Namespace) -> dict:
    return stored_object(save_file(path=args.path, storage=get_storage(), prefix=args.prefix))


def storage_list(args: Namespace) -> list[dict]:
    return [stored_object(value) for value in list_files(storage=get_storage(), prefix=args.prefix)]


def search(args: Namespace) -> list[dict]:
    matches = application.search_documents(application_id=args.application_id, query=args.query, limit=args.limit, metadata_filters=args.metadata_filter, **model_options(args))
    return [dict(embedding_id=str(match.embedding.id), document_id=str(match.document.id),
                 distance=match.distance, content=match.embedding.content,
                 metadata=match.embedding.metadata, storage_key=match.document.storage_key)
            for match in matches]


def embedding_json(value: Embedding) -> dict:
    result = asdict(value)
    for key in ('id', 'document_id'):
        result[key] = str(result[key])
    for key in ('created_at', 'updated_at'):
        result[key] = result[key].isoformat()
    return result


def document_json(value: DocumentDetails) -> dict:
    result = asdict(value.document)
    for key in ('id', 'application_id'):
        result[key] = str(result[key]) if result[key] is not None else None
    for key in ('created_at', 'updated_at'):
        result[key] = result[key].isoformat()
    result['embedding_count'] = len(value.embeddings)
    result['embeddings'] = [dict(id=str(item.id), chunk_index=item.chunk_index,
                                 content=item.content, metadata=item.metadata)
                            for item in sorted(value.embeddings, key=lambda item: (item.chunk_index, item.id))]
    return result


def documents_create(args: Namespace) -> dict:
    return document_json(application.create_document(name=args.name, application_id=args.application_id, path=args.path,
                                                      batch_size=args.batch_size))


def documents_list(args: Namespace) -> list[dict]:
    return [dict(id=str(value.id), name=value.name, embedding_count=value.embedding_count)
            for value in application.enumerate_documents(application_id=args.application_id)]


def documents_show(args: Namespace) -> dict:
    return document_json(application.show_document(document_id=args.id))


def documents_delete(args: Namespace) -> dict:
    return dict(deleted=application.delete_document(document_id=args.id))


def embeddings_show(args: Namespace) -> dict:
    return embedding_json(application.show_embedding(embedding_id=args.id))


def embeddings_delete(args: Namespace) -> dict:
    return dict(deleted=application.delete_embedding(embedding_id=args.id))


def database(args: Namespace) -> dict:
    return database_commands.execute_database_command(args.command)


def application_json(value, *, include_key=False):
    result = asdict(value)
    if not include_key:
        result.pop('api_key')
    result['id'] = str(value.id)
    result['created_at'] = value.created_at.isoformat()
    result['updated_at'] = value.updated_at.isoformat()
    return result


def application_create(args: Namespace) -> dict:
    value = application.create_application(name=args.name, embedding_model=args.embedding_model,
                                          embedding_size=args.embedding_size, provision_type=args.provision_type,
                                          inference_model=args.inference_model, data=args.data,
                                          chunk_size=args.chunk_size, stride=args.stride)
    return application_json(value, include_key=True)


def application_list(args: Namespace) -> list[dict]:
    return [dict(id=str(value.id), name=value.name, provision_type=value.provision_type,
                 embedding_model=value.embedding_model, embedding_size=value.embedding_size,
                 inference_model=value.inference_model, document_count=value.document_count)
            for value in application.enumerate_application_summaries()]


def application_delete(args: Namespace) -> dict:
    return dict(deleted=application.delete_application(application_id=args.application_id))


def application_show(args: Namespace) -> dict:
    return application_json(application.show_application(application_id=args.application_id))


def chat(args: Namespace) -> dict:
    messages = []
    if args.system is not None:
        messages.append(ChatMessage('system', args.system))
    messages.append(ChatMessage('user', args.prompt))
    return asdict(application.chat(messages=messages, application_id=args.application_id,
                                   model_path=args.model, max_tokens=args.max_tokens,
                                   temperature=args.temperature))


def models_list(args: Namespace) -> list[dict]:
    return [asdict(model) for model in application.list_models(
        application_id=args.application_id, discover=args.discover)]


def application_update(args: Namespace) -> dict:
    value = application.show_application(application_id=args.application_id)
    changes = {key: getattr(args, key) for key in ('name', 'embedding_model', 'embedding_size',
               'provision_type', 'inference_model', 'data') if getattr(args, key) is not None}
    if args.clear_inference_model:
        changes['inference_model'] = None
    return application_json(application.update_application(application=replace(value, **changes),
                                                           chunk_size=args.chunk_size, stride=args.stride))


HANDLERS = {
    'applications:delete': application_delete,
    'chat': chat,
    'models:list': models_list,
    'applications:update': application_update,
    'applications:create': application_create,
    'applications:list': application_list,
    'applications:show': application_show,
    'process': process,
    'documents:create': documents_create,
    'documents:show': documents_show,
    'documents:list': documents_list,
    'documents:delete': documents_delete,
    'embeddings:show': embeddings_show,
    'embeddings:delete': embeddings_delete,
    'storage:save': storage_save,
    'storage:list': storage_list,
    'search': search,
    **{name: database for name in
       ('db:create', 'db:migrate', 'db:drop', 'db:reset')},
}


def execute(args: Namespace) -> dict | list:
    return HANDLERS[args.command](args)
