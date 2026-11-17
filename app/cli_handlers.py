"""CLI adapters: map parsed arguments to Python entrypoints and JSON shapes."""
from argparse import Namespace
from dataclasses import asdict

from app import application
from app.database import commands as database_commands
from app.operations.process import process_content
from app.operations.storage import list_files, save_file
from app.operations.text_file import process_text_file
from app.storage.factory import get_storage
from app.storage.models import StoredObject


def model_options(args: Namespace) -> dict:
    return dict(model_path=args.model, context_size=args.context_size, gpu_layers=args.gpu_layers)


def process(args: Namespace) -> dict:
    if args.medium == 'text-file':
        if args.chunk_size is None or args.stride is None:
            raise ValueError('text-file requires --chunk_size and --stride')
        return process_text_file(path=args.content, chunk_size=args.chunk_size, stride=args.stride)
    if args.chunk_size is not None or args.stride is not None:
        raise ValueError('chunking options require text-file')
    return process_content(medium=args.medium, content=args.content)


def embed(args: Namespace) -> dict:
    return application.embed_file(path=args.content, chunk_size=args.chunk_size,
                                  stride=args.stride, batch_size=args.batch_size, **model_options(args))


def stored_object(value: StoredObject) -> dict:
    return dict(key=value.key, size=value.size, content_type=value.content_type, checksum=value.checksum)


def storage_save(args: Namespace) -> dict:
    return stored_object(save_file(path=args.path, storage=get_storage(), prefix=args.prefix))


def storage_list(args: Namespace) -> list[dict]:
    return [stored_object(value) for value in list_files(storage=get_storage(), prefix=args.prefix)]


def search(args: Namespace) -> list[dict]:
    matches = application.search_documents(application_id=args.application_id, query=args.query, limit=args.limit, **model_options(args))
    return [dict(id=str(match.document.id), distance=match.distance,
                 content=match.document.content, storage_key=match.document.storage_key)
            for match in matches]


def ingest(args: Namespace) -> dict:
    documents = application.ingest_file(application_id=args.application_id, path=args.path, chunk_size=args.chunk_size,
                                       stride=args.stride, batch_size=args.batch_size, **model_options(args))
    return dict(indexed_chunks=len(documents), storage_key=documents[0].storage_key if documents else None)


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
                                          inference_model=args.inference_model, data=args.data)
    return application_json(value, include_key=True)


def application_list(args: Namespace) -> list[dict]:
    return [application_json(value) for value in
            application.enumerate_applications(limit=args.limit, offset=args.offset)]


def application_show(args: Namespace) -> dict:
    return application_json(application.show_application(application_id=args.application_id))


HANDLERS = {
    'application:create': application_create,
    'application:list': application_list,
    'application:show': application_show,
    'process': process,
    'embed': embed,
    'storage:save': storage_save,
    'storage:list': storage_list,
    'search': search,
    'ingest': ingest,
    **{name: database for name in
       ('db:create', 'db:migrate', 'db:drop', 'db:reset')},
}


def execute(args: Namespace) -> dict | list:
    return HANDLERS[args.command](args)
