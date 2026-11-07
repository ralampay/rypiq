import argparse
import json

from app.operations.process import process_content
from app.operations.text_file import process_text_file

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

if __name__ == "__main__":
    main()
