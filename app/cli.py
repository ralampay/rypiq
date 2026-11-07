import argparse
import json

from app.operations.process import process_content

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
        choices=["text"],
        help="Content medium",
    )

    process_parser.add_argument(
        "content",
        help="Content to process",
    )

    args = parser.parse_args()

    if args.command == "process":
        result = process_content(
            medium=args.medium,
            content=args.content,
        )

        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
