# Activity 4: Dealing with storage

## Learning outcome

Verify filesystem integration using temporary directories and injected failures.

## Problem and motivation

Retrieval results need a durable reference to the original document. Implement local object storage so operations can use logical keys and later swap to remote storage without rewriting business logic.

## Prerequisites and starter setup

Complete Module 3 and keep its source and tests. Start from that checkpoint, never from the final application.

**Working directory: `~/rag-workshop/`, with `.venv` active.** `WORKSHOP_MATERIALS` is the absolute materials directory exported in Module 1. This overlay changes only this module's files; it preserves earlier completed exercises. It supplies infrastructure and replaces the exercise bodies with TODOs. Commit or copy your previous checkpoint first if you want to compare it later.

```bash
cp -R "$WORKSHOP_MATERIALS/starters/module-04/." .
python -m pip install -e ".[dev]"
```

Read the supplied scaffolding and write the tests below before filling any TODO. The overlay includes those completed test files as an instructor recovery aid; learners can retype or extend them before implementing behavior. Tests from previous modules remain regression checks.

**Files introduced or replaced in this module:** unchanged files from previous modules remain in place. A test may import a newly introduced TODO module and fail; that is intentional Red, not a dependency on a later module.

```text
rag-workshop/
  src/ragworkshop/config.py
  src/ragworkshop/cli.py
  src/ragworkshop/storage/base.py
  src/ragworkshop/storage/local.py
  src/ragworkshop/storage/factory.py
  tests/unit/test_config.py
  tests/integration/test_storage.py
```

## Required tasks

1. Implement the private validation/bundle helpers and all five ObjectStorage methods.
2. Use digest-addressed bundles with payload and metadata.json; stage before publication and remove failed staging directories.
3. Implement get_storage for explicit local adapter selection; reject unknown adapters.
4. Use supplied configuration and CLI wiring without reading environment in LocalStorage.
5. Complete collision, missing-file, unsafe-key, symlink, cleanup, and idempotent-delete tests.
6. Save, retrieve, and process a Module 3 document.

## Exact exercise files and interface stubs

These are the files to create or replace. Bodies marked TODO are deliberately incomplete. All other changed files in the tree are supplied scaffolding, not hidden requirements. The stage overlay contains exactly these stubs and the tests below. Keep public signatures unchanged.

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
        raise NotImplementedError('TODO: LocalStorage._path')

    def _safe(self, path: Path) -> None:
        raise NotImplementedError('TODO: LocalStorage._safe')

    def _object(self, bundle: Path) -> StoredObject:
        raise NotImplementedError('TODO: LocalStorage._object')

    def save(self, source: Path, key: str, content_type: str) -> StoredObject:
        raise NotImplementedError('TODO: LocalStorage.save')

    def get(self, key: str, destination: Path) -> Path:
        raise NotImplementedError('TODO: LocalStorage.get')

    def list(self, prefix: str='') -> list[StoredObject]:
        raise NotImplementedError('TODO: LocalStorage.list')

    def exists(self, key: str) -> bool:
        raise NotImplementedError('TODO: LocalStorage.exists')

    def delete(self, key: str) -> bool:
        raise NotImplementedError('TODO: LocalStorage.delete')
```

### `src/ragworkshop/storage/factory.py`

```python
from pathlib import Path
from ragworkshop.storage.base import ObjectStorage
from ragworkshop.storage.local import LocalStorage

def get_storage(adapter: str, root: Path) -> ObjectStorage:
    raise NotImplementedError('TODO: get_storage')
```

## Red — tests to write or complete

Write these tests before implementing the TODOs. You may use the supplied test files as a starting point; add one edge-case assertion of your own. Do not weaken assertions or mark unfinished behavior skipped.

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

## Green and Refactor constraints

- Implement the private validation/bundle helpers and all five ObjectStorage methods.
- Use digest-addressed bundles with payload and metadata.json; stage before publication and remove failed staging directories.
- Implement get_storage for explicit local adapter selection; reject unknown adapters.
- Use supplied configuration and CLI wiring without reading environment in LocalStorage.
- Complete collision, missing-file, unsafe-key, symlink, cleanup, and idempotent-delete tests.
- Save, retrieve, and process a Module 3 document.
- First satisfy one failing assertion, then rerun the current module tests.
- Refactor only after Green; preserve earlier commands and tests.
- Keep business behavior out of CLI branches. Use supplied interfaces and explicit arguments.
- Keep automated tests offline, with temporary paths for file/database interactions.
- No GPU, cloud credentials, or real GGUF is required.

## Input/output examples and expected observable behavior

**Working directory: `rag-workshop/`.** Output comments specify assertions, not byte-for-byte JSON formatting.

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

## Acceptance criteria

- Implement the private validation/bundle helpers and all five ObjectStorage methods.
- Use digest-addressed bundles with payload and metadata.json; stage before publication and remove failed staging directories.
- Implement get_storage for explicit local adapter selection; reject unknown adapters.
- Use supplied configuration and CLI wiring without reading environment in LocalStorage.
- Complete collision, missing-file, unsafe-key, symlink, cleanup, and idempotent-delete tests.
- Save, retrieve, and process a Module 3 document.
- All provided current-module tests pass without changing the stated interfaces.
- Earlier module regression tests continue to pass.
- Explain one design-pattern participant and its tradeoff to a partner.

## Verification commands

**Working directory: your `rag-workshop/` project root; activate `.venv` first.**

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m pytest -q
rag-workshop info
```

Validated cumulative result: **54 passed, 1 skipped**.

Expected: all implemented tests pass; the real-GGUF test is skipped unless explicitly opted in. This skip is expected, not evidence that real inference was tested.

## Optional stretch challenges

- Design an S3 adapter contract test without making cloud access part of the default suite.
- Add checksum verification on retrieval and a test for externally corrupted payloads.
