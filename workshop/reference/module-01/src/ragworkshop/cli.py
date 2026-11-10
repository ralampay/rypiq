import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-workshop")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info")
    parser.parse_args(argv)
    print("ragworkshop: modular ingestion and retrieval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
