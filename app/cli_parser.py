"""CLI syntax only; parser construction does not create application resources."""
import argparse
import json
from uuid import UUID

from app.embeddings.config import DEFAULT_MODEL


def add_chunk_options(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument('--chunk_size', type=int, required=required,
                        help='Maximum characters per chunk')
    parser.add_argument('--stride', type=int, required=required,
                        help='Characters to advance per chunk')


def add_model_options(parser: argparse.ArgumentParser, *, batching: bool) -> None:
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--context_size', type=int)
    parser.add_argument('--gpu_layers', type=int)
    if batching:
        parser.add_argument('--batch_size', type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='rypiq', description='Rypiq RAG Toolkit')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, description in (
        ('create', 'Create the configured database, vector extension and fixed tables'),
        ('migrate', 'Create missing tables, upgrade legacy documents and validate the schema'),
        ('drop', 'Delete the configured database and all of its data'),
        ('reset', 'Delete and recreate the configured database and fixed tables'),
    ):
        commands.add_parser(f'db:{name}', help=description)

    process = commands.add_parser('process', help='Process content')
    process.add_argument('medium', choices=['text', 'text-file'])
    process.add_argument('content', help='Content to process')
    add_chunk_options(process, required=False)

    embed = commands.add_parser('embed', help='Generate embeddings from a file')
    embed.add_argument('medium', choices=['text-file'])
    embed.add_argument('content', help='Path to a .txt, .pdf, .docx, .xlsx, .xls, .xlsm, or .pptx file')
    add_chunk_options(embed, required=True)
    add_model_options(embed, batching=True)
    embed.add_argument('--output', help='Write full JSON to this path')

    save = commands.add_parser('storage:save', help='Save a file to object storage')
    save.add_argument('path', help='Local path of the file to save')
    save.add_argument('--prefix', default='documents', help='Logical storage prefix')
    listing = commands.add_parser('storage:list', help='List stored files')
    listing.add_argument('--prefix', default='', help='Filter by object key prefix')

    search = commands.add_parser('search', help='Perform semantic vector search')
    search.add_argument('query')
    search.add_argument('--limit', type=int, default=5)
    add_model_options(search, batching=False)

    ingest = commands.add_parser('ingest', help='Store, embed and index a supported document')
    ingest.add_argument('path')
    add_chunk_options(ingest, required=True)
    add_model_options(ingest, batching=True)
    for command in (ingest, search):
        command.add_argument('--application-id', type=UUID, required=True)
        command.set_defaults(model=None, context_size=None, gpu_layers=None)
    create = commands.add_parser('applications:create', help='Create an application')
    create.add_argument('--name', required=True)
    create.add_argument('--embedding-model', required=True)
    create.add_argument('--embedding-size', type=int)
    create.add_argument('--provision-type', choices=['local', 'openai', 'llama-server'], default='local')
    create.add_argument('--inference-model')
    create.add_argument('--data', type=json.loads, default={})
    summaries = commands.add_parser('applications:list', help='List every application ID and name')
    summaries.add_argument('--output', choices=['text', 'json'], default='text',
                           help='Output format (default: text)')
    show = commands.add_parser('applications:show', help='Show application without API key')
    show.add_argument('application_id', type=UUID)
    delete = commands.add_parser('applications:delete', help='Delete an application without documents')
    delete.add_argument('application_id', type=UUID)
    chat = commands.add_parser('chat', help='Run non-streaming text inference')
    chat.add_argument('prompt')
    selection = chat.add_mutually_exclusive_group(required=True)
    selection.add_argument('--application-id', type=UUID)
    selection.add_argument('--model', help='Standalone local GGUF path or @catalog-alias')
    chat.add_argument('--system')
    chat.add_argument('--max-tokens', type=int, default=512)
    chat.add_argument('--temperature', type=float)
    models = commands.add_parser('models:list', help='List catalog models and optionally discover remote models')
    models.add_argument('--application-id', type=UUID)
    models.add_argument('--discover', action='store_true')
    update = commands.add_parser('applications:update', help='Update application configuration')
    update.add_argument('application_id', type=UUID)
    update.add_argument('--name')
    update.add_argument('--embedding-model')
    update.add_argument('--embedding-size', type=int)
    update.add_argument('--provision-type', choices=['local', 'openai', 'llama-server'])
    inference = update.add_mutually_exclusive_group()
    inference.add_argument('--inference-model')
    inference.add_argument('--clear-inference-model', action='store_true')
    update.add_argument('--data', type=json.loads, help='Replace the complete application data JSON object')
    return parser
