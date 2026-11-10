import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ragworkshop.processors.chunks import TextChunker
from ragworkshop.processors.formats import get_processor
from ragworkshop.embeddings.factory import get_embedding_provider
from ragworkshop.operations.embedding import embed_chunks
from ragworkshop.operations.documents import process_document
from ragworkshop.config import load_settings
from ragworkshop.storage.factory import get_storage

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-workshop")
    parser.add_argument("--env", choices=["development", "test", "production"])
    parser.add_argument("--database-config")
    parser.add_argument("--storage-config")
    parser.add_argument("--database-path")
    parser.add_argument("--storage-root")
    parser.add_argument("--embedding-provider", choices=["fake", "local"])
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dimensions", type=int)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info")
    for name in ("process", "embed"):
        command = commands.add_parser(name)
        command.add_argument("text")
        command.add_argument("--format", default="plain")
        command.add_argument("--field", default="text")
        command.add_argument("--chunk-size", type=int, default=200)
        command.add_argument("--stride", type=int, default=150)
        if name == "embed":
            command.add_argument("--provider", choices=["fake", "local"])
            command.add_argument("--model")
            command.add_argument("--dimensions", type=int)
    command = commands.add_parser("process-file")
    command.add_argument("path", type=Path)
    command.add_argument("--chunk-size", type=int, default=200)
    command.add_argument("--stride", type=int, default=150)
    command = commands.add_parser("storage:save")
    command.add_argument("path", type=Path)
    command.add_argument("key")
    command.add_argument("--content-type", default="application/octet-stream")
    command = commands.add_parser("storage:list")
    command.add_argument("--prefix", default="")
    command = commands.add_parser("storage:get")
    command.add_argument("key")
    command.add_argument("destination", type=Path)
    for name in ("storage:exists", "storage:delete"):
        commands.add_parser(name).add_argument("key")
    args = parser.parse_args(argv)
    provider = None
    try:
        if args.command == "info":
            print("ragworkshop: modular ingestion and retrieval")
            return 0
        overrides = {key: value for key, value in vars(args).items() if value is not None}
        for old, new in (("provider", "embedding_provider"), ("model", "embedding_model"),
                         ("dimensions", "embedding_dimensions")):
            if overrides.get(old) is not None:
                overrides[new] = overrides[old]
        settings = load_settings(overrides)
        if args.command in ("process", "embed"):
            processor = get_processor(args.format, TextChunker(args.chunk_size, args.stride), args.field)
            chunks = processor.process(args.text)
            result = {"chunks": [asdict(chunk) for chunk in chunks]}
            if args.command == "embed":
                provider = get_embedding_provider(settings.embedding_provider, settings.embedding_model, settings.embedding_dimensions)
                result["vectors"] = embed_chunks(chunks, provider)
                result["model"] = provider.model_name
        elif args.command == "process-file":
            chunks = process_document(args.path, TextChunker(args.chunk_size, args.stride))
            result = {"chunks": [asdict(chunk) for chunk in chunks]}
        elif args.command.startswith("storage:"):
            storage = get_storage(settings.storage_adapter, settings.storage_root)
            if args.command == "storage:save":
                result = asdict(storage.save(args.path, args.key, args.content_type))
            elif args.command == "storage:list":
                result = [asdict(item) for item in storage.list(args.prefix)]
            elif args.command == "storage:get":
                result = {"destination": str(storage.get(args.key, args.destination))}
            elif args.command == "storage:exists":
                result = {"exists": storage.exists(args.key)}
            else:
                result = {"deleted": storage.delete(args.key)}
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
        return 0
    except ExceptionGroup:
        parser.error("Ingestion failed and cleanup was incomplete; inspect the storage and index")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    finally:
        if provider is not None and hasattr(provider, "close"):
            provider.close()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
