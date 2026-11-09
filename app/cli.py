import argparse
import json
from pathlib import Path

from app.operations.process import process_content
from app.operations.text_file import process_text_file
from app.operations.embedding import (
    DEFAULT_MODEL,
    embed_text_file,
)
from app.operations.storage import (
    save_file,
    list_files,
)

from app.storage.factory import get_storage

from app.database.factory import get_database
from app.database.migrations import migrate

from app.context import create_context

def main():
    parser = argparse.ArgumentParser(
        prog="rypiq",
        description="Rypiq RAG Toolkit",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    subparsers.add_parser(
        "db:create",
        help="Initialize the configured database",
    )

    subparsers.add_parser(
        "db:migrate",
        help="Apply pending database migrations",
    )

    process_parser = subparsers.add_parser(
        "process",
        help="Process content",
    )

    process_parser.add_argument(
        "medium",
        choices=["text", "text-file"],
        help="Content medium",
    )

    process_parser.add_argument(
        "content",
        help="Content to process",
    )

    process_parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Maximuim characters per chunk",
    )

    process_parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Characters to advance per chunk",
    )

    """
    embed_parser arguments
    """
    embed_parser = subparsers.add_parser(
        "embed",
        help="Generate embeddings from a file",
    )

    embed_parser.add_argument(
        "medium",
        choices=["text-file"],
    )

    embed_parser.add_argument(
        "content",
        help="Path to a UTF-8 .txt file",
    )

    embed_parser.add_argument(
        "--chunk_size",
        type=int,
        required=True,
    )
    
    embed_parser.add_argument(
        "--stride",
        type=int,
        required=True,
    )

    embed_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    embed_parser.add_argument(
        "--context_size",
        type=int,
        default=2048,
    )

    embed_parser.add_argument(
        "--gpu_layers",
        type=int,
        default=0,
    )

    embed_parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
    )

    embed_parser.add_argument(
        "--output",
        help="Write full JSON to this path",
    )

    """
    storage_save_parser arguments
    """

    storage_save_parser = subparsers.add_parser(
        "storage:save",
        help="Save a file to object storage",
    )

    storage_save_parser.add_argument(
        "path",
        help="Local path of the file to save",
    )

    storage_save_parser.add_argument(
        "--prefix",
        default="documents",
        help="Logical storage prefix",
    )

    storage_list_parser = subparsers.add_parser(
        "storage:list",
        help="List stored files",
    )

    storage_list_parser.add_argument(
        "--prefix",
        default="",
        help="Filter by object key prefix",
    )

    """
    search_parser arguments
    """

    search_parser = subparsers.add_parser(
        "search",
        help="Perform semantic vector search",
    )

    search_parser.add_argument("query")
    search_parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    if args.command == "process":
        if args.medium == "text-file":
            if args.chunk_size is None or args.stride is None:
                process_parser.error(
                    "text-file requires --chunk_size and --stride"
                )

            try:
                result = process_text_file(
                    path=args.content,
                    chunk_size=args.chunk_size,
                    stride=args.stride,
                )
            except (
                ValueError,
                OSError,
                UnicodeError,
            ) as exc:
                parser.error(str(exc)) 

        else:
            if args.chunk_size is not None or args.stride is not None:
                process_parser.error(
                    "chunking optinos require text-file"
                )

            result = process_content(
                medium=args.medium,
                content=args.content,
            )

        print(json.dumps(result, indent=2))

    elif args.command == "embed":
        try:
            result = embed_text_file(
                path=args.content,
                chunk_size=args.chunk_size,
                stride=args.stride,
                model_path=args.model,
                context_size=args.context_size,
                gpu_layers=args.gpu_layers,
                batch_size=args.batch_size,
            )

            serialized = json.dumps(
                result,
                indent=2,
                allow_nan=False,
            )

            if args.output:
                Path(args.output).write_text(
                    serialized + "\n",
                    encoding="utf-8",
                )
            else:
                print(serialized)
        except(
            ValueError,
            OSError,
            UnicodeError,
            RuntimeError,
        ) as exc:
            embed_parser.error(str(exc))
    elif args.command == "storage:save":
        try:
            storage = get_storage()

            result = save_file(
                path=args.path,
                storage=storage,
                prefix=args.prefix,
            )

            print(json.dumps({
                "key": result.key,
                "size": result.size,
                "content_type": result.content_type,
                "checksum": result.checksum,
            }, indent=2))

        except (ValueError, OSError) as exc:
            storage_save_parser.error(str(exc))

    elif args.command == "storage:list":
        try:
            storage = get_storage()

            objects = list_files(
                storage=storage,
                prefix=args.prefix,
            )

            print(json.dumps([
                {
                    "key": obj.key,
                    "size": obj.size,
                    "content_type": obj.content_type,
                    "checksum": obj.checksum,
                }
                for obj in objects
            ], indent=2))

        except (ValueError, OSError) as exc:
            storage_list_parser.error(str(exc))

    elif args.command == "db:create":
        database = get_database()
        database.create_database()
        print("Database initialized")

    elif args.command == "db:migrate":
        database = get_database()
        completed = migrate(database)

        if not completed:
            print("No pending migrations")
        else:
            for version in completed:
                print(f"Applied migration: {version}")

    elif args.command == "search":
        context = create_context()

        embedding_provider = get_embedding_provider()
        # Use the existing embedding provider factory
        # from your project.

        results = semantic_search(
            query=args.query,
            embedding_provider=embedding_provider,
            repository=context.documents,
            limit=args.limit,
        )

        for result in results:
            print({
                "id": str(result.document.id),
                "distance": result.distance,
                "content": result.document.content,
                "storage_key": result.document.storage_key,
            })

if __name__ == "__main__":
    main()
