# Activity 4 solution: Dealing with storage

**Use this file during review or recovery.**

## Match to the exercise

Retrieval results need a durable reference to the original document. Implement local object storage so operations can use logical keys and later swap to remote storage without rewriting business logic.

This solution uses exactly the activity signatures and requirements. It is a cumulative change: retain the previous module files and replace/add only the files listed here. The direct link below opens the complete cumulative checkpoint for this module.

[Complete Module 4 checkpoint](../../reference/module-04/)

## Red — completed tests first

Run these against the activity stubs and confirm the TODO failures before implementing. These are the same test contracts as Activity.md, with no changed expectations.

### `tests/unit/test_config.py`

```python
from pathlib import Path
import pytest
from ragworkshop.config import load_settings
from ragworkshop.storage.factory import get_storage


def test_precedence_and_dotenv(monkeypatch, tmp_path):
    dotenv = tmp_path / "settings.env"
    dotenv.write_text("RAG_WORKSHOP_ENV=production\nRAG_WORKSHOP_STORAGE_ROOT=/unused\n")
    monkeypatch.setenv("RAG_WORKSHOP_DOTENV", str(dotenv))
    selected = tmp_path / "selected"
    settings = load_settings({"storage_root": selected})
    assert settings.environment == "test"
    assert settings.storage_root == selected
    assert settings.database_path == tmp_path / "db"


def test_yaml_relative_paths_and_isolation(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_WORKSHOP_DATABASE_PATH")
    monkeypatch.delenv("RAG_WORKSHOP_STORAGE_ROOT")
    test = load_settings()
    dev = load_settings({"env": "development"})
    assert test.database_path != dev.database_path
    assert test.storage_root != dev.storage_root
    assert test.storage_root == (Path.cwd() / ".data/test/objects").resolve()
    # Reading settings creates neither environment's persistence directories.


def test_invalid_config(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        load_settings({"env": "typo"})
    bad = tmp_path / "bad.yml"
    bad.write_text("[]")
    with pytest.raises(ValueError, match="section"):
        load_settings({"storage_config": bad})
    with pytest.raises(ValueError, match="Unsupported"):
        get_storage("s3", tmp_path)
```

### `tests/integration/test_storage.py`

```python
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from ragworkshop.storage.local import LocalStorage


def test_round_trip_metadata_collision_and_delete(tmp_path):
    source = tmp_path / "original.txt"
    source.write_bytes(b"python modules")
    storage = LocalStorage(tmp_path / "objects")
    obj = storage.save(source, "documents/lesson.txt", "text/plain")
    assert obj.checksum == hashlib.sha256(b"python modules").hexdigest()
    assert storage.list("documents/") == [obj]
    assert storage.exists(obj.key)
    destination = storage.get(obj.key, tmp_path / "retrieved.txt")
    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        storage.save(source, obj.key, "text/plain")
    with pytest.raises(FileExistsError):
        storage.get(obj.key, destination)
    assert destination.read_bytes() == b"python modules"
    assert storage.delete(obj.key) is True
    assert storage.delete(obj.key) is False
    with pytest.raises(FileNotFoundError):
        storage.get(obj.key, tmp_path / "missing.txt")


@pytest.mark.parametrize("key", ["", "../escape", "/absolute", "a/../b", "a//b", "a/./b", "a\\b"])
def test_unsafe_keys(tmp_path, key):
    storage = LocalStorage(tmp_path / "objects")
    with pytest.raises(ValueError):
        storage.exists(key)


def test_symlink_escape(tmp_path):
    storage = LocalStorage(tmp_path / "objects")
    outside = tmp_path / "outside"
    outside.mkdir()
    bundle = storage.root / hashlib.sha256(b"key").hexdigest()
    bundle.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        storage.exists("key")


def test_save_failure_cleans_staging(monkeypatch, tmp_path):
    import ragworkshop.storage.local as local
    source = tmp_path / "source.txt"
    source.write_text("text")
    storage = LocalStorage(tmp_path / "objects")
    def fail(*args):
        raise OSError("disk failure")
    monkeypatch.setattr(local.os, "rename", fail)
    with pytest.raises(OSError):
        storage.save(source, "key", "text/plain")
    assert list(storage.root.iterdir()) == []
    with pytest.raises(FileNotFoundError):
        storage.save(tmp_path / "missing", "key", "text/plain")


def test_storage_cli_then_document_processing(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("python modules", encoding="utf-8")
    def run(*args):
        result = subprocess.run(["rag-workshop", *map(str, args)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    saved = run("storage:save", source, "documents/source.txt", "--content-type", "text/plain")
    assert run("storage:list")[0] == saved
    destination = tmp_path / "downloaded.txt"
    run("storage:get", saved["key"], destination)
    assert run("process-file", destination)["chunks"][0]["text"] == "python modules"
    assert run("storage:exists", saved["key"]) == {"exists": True}
    assert run("storage:delete", saved["key"]) == {"deleted": True}
```

## Green — complete reference implementation

The exact filename precedes each complete file. Existing earlier-module files not shown here are unchanged. Empty package initializers and non-Python scaffolding are included so the checkpoint is reproducible.

Settings is immutable. The resolver selects and validates environment sections, defines relative-path semantics, and reads dotenv only when explicitly called. Tests override paths before constructing adapters.

### `src/ragworkshop/config.py`

```python
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    environment: str = "development"
    database_adapter: str = "chroma"
    database_path: Path = Path(".data/development/chroma")
    storage_adapter: str = "local"
    storage_root: Path = Path(".data/development/objects")
    embedding_provider: str = "fake"
    embedding_model: str = ""
    embedding_dimensions: int = 8


def read_section(path: Path, environment: str) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError("Invalid YAML configuration") from exc
    if not isinstance(data, dict) or not isinstance(data.get(environment), dict):
        raise ValueError("Missing environment configuration section")
    return data[environment]


def load_settings(overrides: dict[str, Any] | None = None) -> Settings:
    """Call at an entry point only; never from business operations."""
    values = overrides or {}
    load_dotenv(os.getenv("RAG_WORKSHOP_DOTENV", ".env"), override=False)
    def choose(key, default):
        if key in values and values[key] is not None:
            return values[key]
        return os.getenv("RAG_WORKSHOP_" + key.upper(), default)
    environment = choose("env", "development")
    if environment not in {"development", "test", "production"}:
        raise ValueError("Unknown application environment")
    database_file = Path(choose("database_config", "config/database.yml")).expanduser().resolve()
    storage_file = Path(choose("storage_config", "config/storage.yml")).expanduser().resolve()
    database = read_section(database_file, environment)
    storage = read_section(storage_file, environment)
    def resolve(key, section, field, config_file, default):
        explicit = key in values or "RAG_WORKSHOP_" + key.upper() in os.environ
        raw = choose(key, section.get(field, default))
        if not isinstance(raw, (str, Path)) or not str(raw).strip():
            raise ValueError(f"Invalid path setting: {key}")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (Path.cwd() if explicit else config_file.parent) / path
        return path.resolve()
    embedding = database.get("embedding", {})
    if not isinstance(embedding, dict):
        raise ValueError("embedding must be a mapping")
    dimensions = int(choose("embedding_dimensions", embedding.get("dimensions", 8)))
    if dimensions < 1:
        raise ValueError("Embedding dimensions must be positive")
    return Settings(
        environment=environment,
        database_adapter=choose("database_adapter", database.get("adapter", "chroma")),
        database_path=resolve("database_path", database, "path", database_file,
                              f"../.data/{environment}/chroma"),
        storage_adapter=choose("storage_adapter", storage.get("adapter", "local")),
        storage_root=resolve("storage_root", storage, "root", storage_file,
                             f"../.data/{environment}/objects"),
        embedding_provider=choose("embedding_provider", embedding.get("provider", "fake")),
        embedding_model=choose("embedding_model", embedding.get("model", "")),
        embedding_dimensions=dimensions,
    )
```

main accepts argv for direct testing; argparse parses input and formats usage errors. JSON serialization is presentation logic. Command handlers delegate behavior and close providers they construct.

### `src/ragworkshop/cli.py`

```python
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
```

The Protocol specifies behavior without requiring inheritance. Implementations and lightweight fakes must preserve this contract. Validation belongs at boundaries so malformed provider output never reaches persistence.

### `src/ragworkshop/storage/base.py`

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class StoredObject:
    key: str
    size: int
    content_type: str
    checksum: str


class ObjectStorage(Protocol):
    def save(self, source: Path, key: str, content_type: str) -> StoredObject: ...
    def get(self, key: str, destination: Path) -> Path: ...
    def list(self, prefix: str = "") -> list[StoredObject]: ...
    def delete(self, key: str) -> bool: ...
    def exists(self, key: str) -> bool: ...
```

The concrete adapter owns external resource handling and validation. Its private helpers centralize invariants so public methods remain small. Inspect the module-specific discussion for the atomic-write or model-response policy.

### `src/ragworkshop/storage/local.py`

```python
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from ragworkshop.storage.base import StoredObject


class LocalStorage:
    """Single-process adapter; root must not be writable by untrusted users."""
    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        if (not key or "\\" in key or "\x00" in key or key.startswith("/")
                or any(part in ("", ".", "..") for part in key.split("/"))):
            raise ValueError("Unsafe storage key")
        # A digest gives every key one independent bundle, avoiding prefix collisions.
        return self.root / hashlib.sha256(key.encode("utf-8")).hexdigest()

    def _safe(self, path: Path) -> None:
        if path.is_symlink() or not path.resolve().is_relative_to(self.root):
            raise ValueError("Unsafe storage symlink")

    def _object(self, bundle: Path) -> StoredObject:
        self._safe(bundle)
        self._safe(bundle / "metadata.json")
        self._safe(bundle / "payload")
        return StoredObject(**json.loads((bundle / "metadata.json").read_text(encoding="utf-8")))

    def save(self, source: Path, key: str, content_type: str) -> StoredObject:
        bundle = self._path(key)
        self._safe(bundle)
        source = Path(source).expanduser().resolve(strict=True)
        if not source.is_file():
            raise ValueError("Source must be a regular file")
        if bundle.exists():
            raise FileExistsError(key)
        if not content_type.strip():
            raise ValueError("Content type must not be empty")
        staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=self.root))
        try:
            payload = staging / "payload"
            shutil.copyfile(source, payload)
            with payload.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            stored = StoredObject(key, payload.stat().st_size, content_type, checksum)
            (staging / "metadata.json").write_text(json.dumps(asdict(stored)), encoding="utf-8")
            # Completed bundles are nonempty directories: rename cannot replace one.
            os.rename(staging, bundle)
            return stored
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def get(self, key: str, destination: Path) -> Path:
        bundle = self._path(key)
        self._object(bundle)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        created = False
        try:
            with destination.open("xb") as writer:
                created = True
                with (bundle / "payload").open("rb") as reader:
                    shutil.copyfileobj(reader, writer)
        except Exception:
            if created:
                destination.unlink(missing_ok=True)
            raise
        return destination

    def list(self, prefix: str = "") -> list[StoredObject]:
        if prefix:
            self._path(prefix.rstrip("/"))
        objects = [self._object(bundle) for bundle in self.root.iterdir()
                   if not bundle.name.startswith(".pending-")]
        return sorted((obj for obj in objects if obj.key.startswith(prefix)), key=lambda obj: obj.key)

    def exists(self, key: str) -> bool:
        bundle = self._path(key)
        self._safe(bundle)
        if not bundle.exists():
            return False
        self._object(bundle)
        return True

    def delete(self, key: str) -> bool:
        if not self.exists(key):
            return False
        shutil.rmtree(self._path(key))
        return True
```

A simple factory constructs a supported adapter using explicit arguments and rejects unknown names. It does not reach into global settings.

### `src/ragworkshop/storage/factory.py`

```python
from pathlib import Path
from ragworkshop.storage.base import ObjectStorage
from ragworkshop.storage.local import LocalStorage


def get_storage(adapter: str, root: Path) -> ObjectStorage:
    if adapter != "local":
        raise ValueError(f"Unsupported storage adapter: {adapter}")
    return LocalStorage(root)
```

## Refactor, pattern rationale, and alternatives

A file plus sidecar metadata is simpler but needs a publication protocol to avoid mismatched visibility. Bundles solve that here at the cost of hidden internal layout. S3 could implement the same logical interface, but its pagination, conditional writes, and metadata semantics require adapter-specific tests.

Object storage preserves original bytes. A vector database indexes chunk vectors and metadata. Embedding computes vectors. These are separate responsibilities even when all three run locally. `StoredObject` contains a logical key, byte size, content type, and SHA-256 checksum; it deliberately exposes no local payload path. A caller retrieves through `get`, which makes a future remote object-store adapter possible.

The ObjectStorage Protocol defines save/get/list/delete/exists. `LocalStorage` adapts files and directories to this contract; `get_storage` is the simple factory. Configuration selects the adapter and root, but business operations receive an already-built object. Replacing local storage must not change an ingestion algorithm that uses only the contract.

Keys are slash-separated logical names. A namespace such as `documents/` groups objects; listing uses literal string-prefix matching, so `doc` also matches `documents/`. Empty keys, absolute paths, traversal components, empty components, backslashes, and NUL are rejected. Prefixes may end in one slash. UUID keys in ingestion avoid accidental filename collisions; direct repeated keys raise `FileExistsError` instead of overwriting data.

Internally the adapter hashes a validated key to a directory name. Each bundle holds `payload` and `metadata.json`. Logical keys remain in metadata. This avoids collisions between a key and another key's metadata filename, and allows `a` and `a/b` to coexist. A temporary sibling bundle is completely populated before an atomic rename publishes it; list ignores staging bundles. Content type and checksum are therefore stable between save and list. Atomic visibility is not a guarantee of durability after power loss; production storage may need fsync and recovery procedures.

The root is trusted and owned by the application. The adapter rejects bundle and payload symlinks, but is not a defense against an attacker concurrently replacing the root directory. It is a single-process teaching adapter, not a multi-writer storage service. `get` uses exclusive destination creation and removes only its own partially written destination on failure. Missing get raises, missing exists is false, and repeated delete returns false. Cleanup must never delete a pre-existing destination.

Settings resolution is supplied scaffolding here and studied fully in Module 5. Read YAML with `safe_load`, validate environment sections, and resolve relative YAML paths relative to the YAML file. CLI/environment path overrides resolve from the working directory. No configuration is loaded during import.

## Verification and expected test outcomes

**Working directory: your `rag-workshop/` project root; activate `.venv` first.**

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m pytest -q
rag-workshop info
```

Validated cumulative result: **54 passed, 1 skipped**.

Expected: all implemented tests pass; the real-GGUF test is skipped unless explicitly opted in. This skip is expected, not evidence that real inference was tested.

**Working directory: `rag-workshop/`.**

```bash
rag-workshop --env test storage:save samples/sample.txt documents/sample.txt --content-type text/plain
# JSON: key documents/sample.txt, size 42, content_type text/plain, SHA-256 checksum
rag-workshop --env test storage:list --prefix documents/
# array containing the same StoredObject metadata
rag-workshop --env test storage:exists documents/sample.txt
# {"exists": true}
rag-workshop --env test storage:get documents/sample.txt samples/retrieved.txt
# {"destination": "samples/retrieved.txt"}
rag-workshop process-file samples/retrieved.txt
# original text and source metadata
rag-workshop --env test storage:delete documents/sample.txt
# {"deleted": true}; repeating this command returns {"deleted": false}
```

## Edge cases and limitations

Listing a namespace preserves logical keys even though disk directories use hashes. A key and its child prefix can coexist. get must never overwrite an existing path or delete it on failure. Symlink checks assume a trusted root, not hostile concurrent modification. Atomic bundle publication provides visibility, not crash durability or distributed transactions.

## Why the acceptance criteria are met

- Implement the private validation/bundle helpers and all five ObjectStorage methods. The completed code and assertions above make this behavior observable.
- Use digest-addressed bundles with payload and metadata.json; stage before publication and remove failed staging directories. The completed code and assertions above make this behavior observable.
- Implement get_storage for explicit local adapter selection; reject unknown adapters. The completed code and assertions above make this behavior observable.
- Use supplied configuration and CLI wiring without reading environment in LocalStorage. The completed code and assertions above make this behavior observable.
- Complete collision, missing-file, unsafe-key, symlink, cleanup, and idempotent-delete tests. The completed code and assertions above make this behavior observable.
- Save, retrieve, and process a Module 3 document. The completed code and assertions above make this behavior observable.

Tests are the executable acceptance evidence; successful imports alone are insufficient. Exact validation counts for all checkpoints are recorded in the workshop README. The optional real-model check is reported separately from default offline validation.
