# Introduction

Build a file-storage and ingestion workflow for protocol documents. Preserve the original file, generate embeddings for its chunks, and insert those chunks into Chroma.

Complete [EmbeddingGeneration.md](EmbeddingGeneration.md) and [DatabaseSetup.md](DatabaseSetup.md) first. Keep their modules and tests in the same `protocol_embeddings` tutorial package.

# Objectives

- Test local file storage before implementing it.
- Connect chunking, embedding, file storage, and database insertion.
- Clean up records and files when an insertion fails.
- Add CLI utilities for saving files, ingesting protocols, and listing results.

# Workflow

Use the activated tutorial environment with `pytest` and `chromadb==1.5.9` installed. Work from the directory containing `protocol_embeddings`. Save and run each section's tests before adding its implementation.

There are two kinds of listing: `storage:list` returns original files; `documents:list` returns indexed chunk records. Saving a file alone does not index it.

## Section 1

Build local file storage with unique keys and checksums.

### Tests to Write

Save as `tests/unit/test_storage_operation.py`:

```python
from pathlib import Path
from unittest.mock import Mock

import pytest

from protocol_embeddings.storage import save_file


def test_save_generates_distinct_keys(tmp_path):
    source = tmp_path / "protocol.TXT"
    source.write_text("Measure", encoding="utf-8")
    storage = Mock()
    save_file(source, storage)
    save_file(source, storage)
    first, second = storage.save.call_args_list
    assert first.args[0] == source
    assert first.args[1].startswith("documents/")
    assert Path(first.args[1]).suffix == ".txt"
    assert first.args[1] != second.args[1]


def test_missing_source_never_calls_storage(tmp_path):
    storage = Mock()
    with pytest.raises(FileNotFoundError):
        save_file(tmp_path / "missing.txt", storage)
    storage.save.assert_not_called()
```

Save as `tests/integration/test_storage.py`:

```python
import hashlib

import pytest

from protocol_embeddings.storage import LocalStorage


def test_save_list_delete(tmp_path):
    source = tmp_path / "protocol.txt"
    source.write_bytes(b"Record sample ID")
    root = tmp_path / "files"
    storage = LocalStorage(root)
    assert storage.list() == []
    stored = storage.save(source, "documents/source.txt")
    assert stored.size == len(source.read_bytes())
    assert stored.checksum == hashlib.sha256(source.read_bytes()).hexdigest()
    assert (root / stored.key).read_bytes() == source.read_bytes()
    assert LocalStorage(root).list() == [stored]
    with pytest.raises(FileExistsError):
        storage.save(source, stored.key)
    assert (root / stored.key).read_bytes() == source.read_bytes()
    assert storage.delete(stored.key) is True
    assert storage.delete(stored.key) is False


@pytest.mark.parametrize("key", ["", "../escape.txt", "/absolute.txt", "a/../b", "a\\b"])
def test_invalid_storage_keys(tmp_path, key):
    source = tmp_path / "source.txt"
    source.write_text("text")
    with pytest.raises(ValueError):
        LocalStorage(tmp_path / "files").save(source, key)


def test_symlink_escape(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    (root / "outside").symlink_to(tmp_path, target_is_directory=True)
    source = tmp_path / "source.txt"
    source.write_text("text")
    with pytest.raises(ValueError):
        LocalStorage(root).save(source, "outside/copy.txt")
```

Run `python -m pytest -q tests/unit/test_storage_operation.py tests/integration/test_storage.py`.

### Implementation

Use exclusive file creation to prevent overwrites. Remove a partially copied file if copying fails. The storage directory is intended for a trusted, single-writer tutorial application.

Save as `protocol_embeddings/storage.py`:

```python
import hashlib
import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class StoredObject:
    key: str
    size: int
    checksum: str
    content_type: str


class LocalStorage:
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()

    def _resolve(self, key):
        if (not key or "\\" in key or key.startswith("/")
                or any(part in ("", ".", "..") for part in key.split("/"))):
            raise ValueError("Invalid storage key")
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("Storage key escapes root")
        return path

    def _describe(self, key, path):
        digest = hashlib.sha256()
        with path.open("rb") as reader:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                digest.update(block)
        return StoredObject(
            key=key, size=path.stat().st_size, checksum=digest.hexdigest(),
            content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
        )

    def save(self, source, key):
        source = Path(source)
        if not source.is_file():
            raise FileNotFoundError(source)
        target = self._resolve(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as reader:
            writer = target.open("xb")  # A collision must not delete an earlier file.
            try:
                with writer:
                    shutil.copyfileobj(reader, writer)
                return self._describe(key, target)
            except Exception:
                target.unlink(missing_ok=True)
                raise

    def list(self):
        if not self.root.exists():
            return []
        return [self._describe(path.relative_to(self.root).as_posix(), path)
                for path in sorted(self.root.rglob("*"))
                if path.is_file() and not path.is_symlink()]

    def delete(self, key):
        path = self._resolve(key)
        if not path.is_file():
            return False
        path.unlink()
        return True


def save_file(path, storage):
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    key = f"documents/{uuid4()}{source.suffix.lower()}"
    return storage.save(source, key)
```

## Section 2

Compose the ingestion operation and test cleanup on failure.

### Tests to Write

Save as `tests/unit/test_ingestion.py`:

```python
from unittest.mock import Mock

import pytest

from protocol_embeddings.ingestion import ingest_document
from protocol_embeddings.storage import StoredObject


@pytest.fixture
def dependencies(tmp_path, monkeypatch):
    source = tmp_path / "protocol.txt"
    source.write_text("ABCDEFG", encoding="utf-8")
    embed = Mock(return_value={"model": "fake.gguf", "chunks": [
        {"index": 0, "start": 0, "end": 4, "text": "ABCD", "embedding": [1.0, 0.0]},
        {"index": 1, "start": 3, "end": 7, "text": "DEFG", "embedding": [0.0, 1.0]},
    ]})
    monkeypatch.setattr("protocol_embeddings.ingestion.embed_text_file", embed)
    storage = Mock()
    storage.save.return_value = StoredObject("documents/source.txt", 7, "checksum", "text/plain")
    repository = Mock()
    repository.create.side_effect = lambda document: document
    return source, embed, storage, repository


def test_ingest_preserves_source_and_offsets(dependencies):
    source, embed, storage, repository = dependencies
    events = Mock()
    events.attach_mock(embed, "embed")
    events.attach_mock(storage.save, "save")
    events.attach_mock(repository.create, "create")
    documents = ingest_document(source, storage, repository, 4, 3, "fake.gguf")
    assert [event[0] for event in events.mock_calls] == ["embed", "save", "create", "create"]
    assert documents[0].source_id == documents[1].source_id
    assert documents[0].storage_key == "documents/source.txt"
    assert documents[1].start == 3
    assert documents[1].end == 7
    assert documents[1].embedding_model == "fake.gguf"


def test_embedding_failure_saves_nothing(dependencies):
    source, embed, storage, repository = dependencies
    embed.side_effect = ValueError("bad input")
    with pytest.raises(ValueError):
        ingest_document(source, storage, repository, 4, 3, "fake.gguf")
    storage.save.assert_not_called()
    repository.create.assert_not_called()


def test_insertion_failure_cleans_up(dependencies):
    source, _, storage, repository = dependencies
    repository.create.side_effect = [None, RuntimeError("insert failed")]
    with pytest.raises(RuntimeError, match="insert failed"):
        ingest_document(source, storage, repository, 4, 3, "fake.gguf")
    # Include the attempted insert: it may have written before raising.
    attempted_ids = [call.args[0].id for call in repository.create.call_args_list]
    assert [call.args[0] for call in repository.delete.call_args_list] == attempted_ids[::-1]
    storage.delete.assert_called_once_with("documents/source.txt")


def test_cleanup_continues_if_record_deletion_fails(dependencies):
    source, _, storage, repository = dependencies
    repository.create.side_effect = [None, RuntimeError("insert failed")]
    repository.delete.side_effect = OSError("delete failed")
    with pytest.raises(RuntimeError, match="cleanup was incomplete") as error:
        ingest_document(source, storage, repository, 4, 3, "fake.gguf")
    assert str(error.value.__cause__) == "insert failed"
    assert repository.delete.call_count == 2
    storage.delete.assert_called_once()
```

Run `python -m pytest -q tests/unit/test_ingestion.py`.

### Implementation

Generate all embeddings before saving the original. Give every ingestion a new source ID. Track attempted inserts so cleanup also handles an insert that wrote a record before raising.

Save as `protocol_embeddings/ingestion.py`:

```python
from pathlib import Path
from uuid import uuid4

from protocol_embeddings.documents import Document
from protocol_embeddings.operations import embed_text_file
from protocol_embeddings.storage import save_file


def ingest_document(path, storage, repository, chunk_size, stride, model_path,
                    batch_size=1, provider_factory=None):
    source = Path(path)
    result = embed_text_file(
        str(source), chunk_size, stride, model_path,
        batch_size=batch_size, provider_factory=provider_factory,
    )
    stored = save_file(source, storage)
    source_id = str(uuid4())
    attempted = []
    documents = []
    try:
        for chunk in result["chunks"]:
            document = Document(
                content=chunk["text"], embedding=chunk["embedding"],
                source_id=source_id, storage_key=stored.key,
                filename=source.name, source=str(source),
                chunk_index=chunk["index"], start=chunk["start"], end=chunk["end"],
                embedding_model=result["model"],
            )
            attempted.append(document.id)
            repository.create(document)
            documents.append(document)
    except Exception as error:
        cleanup_errors = []
        for record_id in reversed(attempted):
            try:
                repository.delete(record_id)
            except Exception as cleanup_error:
                cleanup_errors.append(cleanup_error)
        try:
            storage.delete(stored.key)
        except Exception as cleanup_error:
            cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            raise RuntimeError("Ingestion failed; cleanup was incomplete") from error
        raise
    return documents
```

Cleanup is best effort, not a transaction spanning Chroma and the filesystem. A crash or cleanup error may leave data requiring inspection. Re-ingesting the same path creates a new source and new records; this example does not deduplicate files.

## Section 3

Add CLI commands for storage, ingestion, and document listing.

### Tests to Write

Save as `tests/integration/test_ingestion_commands.py`:

```python
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from protocol_embeddings.commands import main
from protocol_embeddings.database import Database, migrate


def test_ingest_and_list_across_processes(tmp_path, monkeypatch, capsys):
    db_path = tmp_path / "chroma"
    files = tmp_path / "files"
    source = tmp_path / "protocol.txt"
    source.write_text("ABCDEFGHIJ", encoding="utf-8")
    model = tmp_path / "fake.gguf"
    model.touch()
    migrate(Database(db_path))
    engine = Mock()
    engine.create_embedding.side_effect = lambda texts: {"data": [
        {"index": i, "embedding": [1.0, 0.0]} for i, _ in enumerate(texts)
    ]}
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=Mock(return_value=engine)))
    options = ["--db-path", str(db_path), "--storage-root", str(files)]
    main(options + ["ingest", str(source), "--model", str(model),
                    "--chunk-size", "4", "--stride", "3"])
    report = json.loads(capsys.readouterr().out)
    assert report["indexed_chunks"] == 3
    assert (files / report["storage_key"]).read_bytes() == source.read_bytes()

    command = [sys.executable, "-m", "protocol_embeddings.commands"] + options
    output = subprocess.run(command + ["documents:list", "--limit", "2"],
                            check=True, capture_output=True, text=True)
    documents = json.loads(output.stdout)
    assert len(documents) == 2
    assert all("embedding" not in document for document in documents)
    assert len({document["source_id"] for document in documents}) == 1
    assert all(document["storage_key"] == report["storage_key"] for document in documents)
    tail = subprocess.run(command + ["documents:list", "--offset", "2"],
                          check=True, capture_output=True, text=True)
    assert len(json.loads(tail.stdout)) == 1
    output = subprocess.run(command + ["storage:list"], check=True, capture_output=True, text=True)
    assert len(json.loads(output.stdout)) == 1


def test_storage_save_does_not_index(tmp_path, capsys):
    source = tmp_path / "protocol.txt"
    source.write_text("Measure", encoding="utf-8")
    database = Database(tmp_path / "chroma")
    migrate(database)
    main(["--db-path", str(database.path), "--storage-root", str(tmp_path / "files"),
          "storage:save", str(source)])
    assert json.loads(capsys.readouterr().out)["size"] == 7
    assert database.documents().list() == []


def test_list_rejects_invalid_pagination(tmp_path, capsys):
    database = Database(tmp_path / "chroma")
    migrate(database)
    with pytest.raises(SystemExit) as error:
        main(["--db-path", str(database.path), "documents:list", "--limit", "0"])
    assert error.value.code == 2
    assert "Invalid pagination" in capsys.readouterr().err
```

Run `python -m pytest -q tests/integration/test_ingestion_commands.py`. Only inference is substituted; these tests use real local storage and a real persistent Chroma database.

### Implementation

Replace `protocol_embeddings/commands.py` with this complete version. It preserves the database commands and adds storage and ingestion commands.

Save as `protocol_embeddings/commands.py`:

```python
import argparse
import json

from chromadb.errors import ChromaError

from protocol_embeddings.database import Database, migrate
from protocol_embeddings.storage import LocalStorage, save_file
from protocol_embeddings.ingestion import ingest_document


def main(argv=None):
    parser = argparse.ArgumentParser(description="Protocol document tools")
    parser.add_argument("--db-path", default="data/chroma")
    parser.add_argument("--storage-root", default="data/files")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("db:create")
    sub.add_parser("db:migrate")
    save = sub.add_parser("storage:save")
    save.add_argument("path")
    sub.add_parser("storage:list")
    ingest = sub.add_parser("ingest")
    ingest.add_argument("path")
    ingest.add_argument("--model", required=True)
    ingest.add_argument("--chunk-size", type=int, default=500)
    ingest.add_argument("--stride", type=int, default=400)
    ingest.add_argument("--batch-size", type=int, default=1)
    listing = sub.add_parser("documents:list")
    listing.add_argument("--limit", type=int, default=100)
    listing.add_argument("--offset", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        database = Database(args.db_path)
        if args.command == "db:create":
            database.create_database()
            result = {"database": str(database.path)}
        elif args.command == "db:migrate":
            result = {"applied": migrate(database)}
        elif args.command == "storage:save":
            stored = save_file(args.path, LocalStorage(args.storage_root))
            result = vars(stored)
        elif args.command == "storage:list":
            result = [vars(item) for item in LocalStorage(args.storage_root).list()]
        elif args.command == "ingest":
            documents = ingest_document(
                args.path, LocalStorage(args.storage_root), database.documents(),
                args.chunk_size, args.stride, args.model, batch_size=args.batch_size,
            )
            result = {"indexed_chunks": len(documents),
                      "storage_key": documents[0].storage_key}
        elif args.command == "documents:list":
            result = [document.summary() for document in
                      database.documents().list(args.limit, args.offset)]
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, RuntimeError, ChromaError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
```

Run the complete test suite, then try the workflow with a real model:

```bash
python -m pytest -q tests
python -m pip install llama-cpp-python
python -m protocol_embeddings.commands --db-path data/chroma db:create
python -m protocol_embeddings.commands --db-path data/chroma db:migrate
cat > protocol.txt <<'TEXT'
Record the sample identifier.
Check instrument calibration before processing.
Save the measurement and its timestamp.
TEXT

python -m protocol_embeddings.commands --db-path data/chroma --storage-root data/files \
  ingest protocol.txt --model models/embeddinggemma-300m-Q4_0.gguf \
  --chunk-size 500 --stride 400
python -m protocol_embeddings.commands --storage-root data/files storage:list
python -m protocol_embeddings.commands --db-path data/chroma documents:list --limit 10 --offset 0
```

Provide the GGUF model at the path shown, or replace the argument with your model path. This is an optional real-inference check; automated integration tests need no weights. `ingest` saves the original itself, so do not first call `storage:save` for that same workflow.

To save a file without indexing it:

```bash
python -m protocol_embeddings.commands --storage-root data/files storage:save protocol.txt
```

Document listings include source IDs, storage keys, text, chunk indices, offsets, and model names. They omit vector arrays for readability. Listing is paginated but is not a similarity ranking. Continue with [SimpleQuery.md](SimpleQuery.md).

