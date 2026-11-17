"""CLI syntax only; parser construction does not create application resources."""
import argparse

from app.embeddings.config import DEFAULT_MODEL


def add_chunk_options(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument('--chunk_size', type=int, required=required,
                        help='Maximum characters per chunk')
    parser.add_argument('--stride', type=int, required=required,
                        help='Characters to advance per chunk')


def add_model_options(parser: argparse.ArgumentParser, *, batching: bool) -> None:
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--context_size', type=int, default=2048)
    parser.add_argument('--gpu_layers', type=int, default=0)
    if batching:
        parser.add_argument('--batch_size', type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='rypiq', description='Rypiq RAG Toolkit')
    commands = parser.add_subparsers(dest='command', required=True)
    for name, description in (
        ('create', 'Create the configured database, vector extension and fixed tables'),
        ('migrate', 'Create missing tables and validate the fixed schema'),
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
    embed.add_argument('content', help='Path to a UTF-8 .txt file')
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

    ingest = commands.add_parser('ingest', help='Store, embed and index a text file')
    ingest.add_argument('path')
    add_chunk_options(ingest, required=True)
    add_model_options(ingest, batching=True)
    return parser
