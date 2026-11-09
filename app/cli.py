import argparse
import json
from pathlib import Path

from app.operations.process import process_content
from app.operations.text_file import process_text_file
from app.operations.embedding import (
    DEFAULT_MODEL,
    embed_text_file,
)

def main():
    parser = argparse.ArgumentParser(
        prog="rypiq",
        description="Rypiq RAG Toolkit",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
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

if __name__ == "__main__":
    main()
