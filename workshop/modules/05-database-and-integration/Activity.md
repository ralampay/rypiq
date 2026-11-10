# Activity 5: Database, configuration, and end-to-end integration

## Learning outcome

Prove persistence, environment isolation, and compensation using deterministic E2E tests.

## Problem and motivation

Users need to ingest a document, restart the program, and retrieve relevant chunks with their source information. Compose the existing components with persistent Chroma and prove this works without a model download.

## Prerequisites and starter setup

Complete Module 4 and keep its source and tests. Start from that checkpoint, never from the final application.

**Working directory: `~/rag-workshop/`, with `.venv` active.** `WORKSHOP_MATERIALS` is the absolute materials directory exported in Module 1. This overlay changes only this module's files; it preserves earlier completed exercises. It supplies infrastructure and replaces the exercise bodies with TODOs. Commit or copy your previous checkpoint first if you want to compare it later.

```bash
cp -R "$WORKSHOP_MATERIALS/starters/module-05/." .
python -m pip install -e ".[dev]"
```

Read the supplied scaffolding and write the tests below before filling any TODO. The overlay includes those completed test files as an instructor recovery aid; learners can retype or extend them before implementing behavior. Tests from previous modules remain regression checks.

**Files introduced or replaced in this module:** unchanged files from previous modules remain in place. A test may import a newly introduced TODO module and fail; that is intentional Red, not a dependency on a later module.

```text
rag-workshop/
  src/ragworkshop/context.py
  src/ragworkshop/cli.py
  src/ragworkshop/database/models.py
  src/ragworkshop/database/base.py
  src/ragworkshop/database/chroma.py
  src/ragworkshop/database/factory.py
  src/ragworkshop/database/migrations.py
  src/ragworkshop/operations/ingestion.py
  src/ragworkshop/operations/search.py
  tests/unit/test_ingestion.py
  tests/integration/test_database.py
  tests/e2e/test_pipeline.py
```

## Required tasks

1. Implement ChromaDocumentRepository against the supplied backend and record types.
2. Validate the model identity and dimensions before indexing/search; use explicit vectors.
3. Implement create_context with an optional injected provider and deterministic ownership rules.
4. Implement ingest_document: store, extract/chunk, embed, persist, and compensate partial failures.
5. Implement semantic_search with validation, same-space query embedding, filters, and source-bearing results.
6. Use .env for the manual demo and monkeypatch temporary paths for automated tests.
7. Verify migrations, fresh backend persistence, process restart persistence, environment isolation, and all three document formats.

## Exact exercise files and interface stubs

These are the files to create or replace. Bodies marked TODO are deliberately incomplete. All other changed files in the tree are supplied scaffolding, not hidden requirements. The stage overlay contains exactly these stubs and the tests below. Keep public signatures unchanged.

### `src/ragworkshop/database/chroma.py`

```python
from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from ragworkshop.database.models import Document, SearchResult
from ragworkshop.embeddings.base import validate_vectors
from ragworkshop.processors.chunks import Metadata

class ChromaDocumentRepository:

    def __init__(self, collection):
        self.collection = collection
        self.dimensions: int | None = None

    def ensure_space(self, model_name: str, dimensions: int) -> None:
        raise NotImplementedError('TODO: ChromaDocumentRepository.ensure_space')

    def add(self, documents: list[Document]) -> None:
        raise NotImplementedError('TODO: ChromaDocumentRepository.add')

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError('TODO: ChromaDocumentRepository.delete')

    def search_by_vector(self, embedding: list[float], limit: int=5, filters: Metadata | None=None) -> list[SearchResult]:
        raise NotImplementedError('TODO: ChromaDocumentRepository.search_by_vector')

class ChromaDatabaseBackend:

    def __init__(self, path: Path):
        self.path = Path(path)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.path), settings=ChromaSettings(anonymized_telemetry=False))
        return self._client

    def create_database(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        _ = self.client

    def create_documents_collection(self) -> None:
        self.client.get_or_create_collection('documents', embedding_function=None, configuration={'hnsw': {'space': 'cosine'}})

    def _migrations(self):
        return self.client.get_or_create_collection('ragworkshop_migrations', embedding_function=None)

    def applied_migrations(self) -> set[str]:
        return set(self._migrations().get(include=[])['ids'])

    def record_migration(self, version: str) -> None:
        self._migrations().upsert(ids=[version], embeddings=[[1.0]], documents=[version])

    def documents(self) -> ChromaDocumentRepository:
        return ChromaDocumentRepository(self.client.get_collection('documents', embedding_function=None))
```

### `src/ragworkshop/context.py`

```python
from contextlib import contextmanager
from dataclasses import dataclass
from collections.abc import Iterator
from ragworkshop.config import Settings
from ragworkshop.database.base import DocumentRepository
from ragworkshop.database.factory import get_database
from ragworkshop.database.migrations import migrate
from ragworkshop.embeddings.base import EmbeddingProvider
from ragworkshop.embeddings.factory import get_embedding_provider
from ragworkshop.storage.base import ObjectStorage
from ragworkshop.storage.factory import get_storage

@dataclass(frozen=True)
class ApplicationContext:
    documents: DocumentRepository
    storage: ObjectStorage
    embeddings: EmbeddingProvider

@contextmanager
def create_context(settings: Settings, provider: EmbeddingProvider | None=None) -> Iterator[ApplicationContext]:
    raise NotImplementedError('TODO: create_context')
```

### `src/ragworkshop/operations/ingestion.py`

```python
import mimetypes
from pathlib import Path
from uuid import uuid4
from ragworkshop.context import ApplicationContext
from ragworkshop.database.models import Document
from ragworkshop.operations.documents import process_document
from ragworkshop.operations.embedding import embed_chunks
from ragworkshop.processors.chunks import TextChunker

def ingest_document(path: Path, context: ApplicationContext, chunker: TextChunker) -> list[Document]:
    raise NotImplementedError('TODO: ingest_document')
```

### `src/ragworkshop/operations/search.py`

```python
from ragworkshop.context import ApplicationContext
from ragworkshop.database.models import SearchResult
from ragworkshop.embeddings.base import validate_vectors
from ragworkshop.processors.chunks import Metadata

def semantic_search(query: str, context: ApplicationContext, limit: int=5, filters: Metadata | None=None) -> list[SearchResult]:
    raise NotImplementedError('TODO: semantic_search')
```

## Red — tests to write or complete

Write these tests before implementing the TODOs. You may use the supplied test files as a starting point; add one edge-case assertion of your own. Do not weaken assertions or mark unfinished behavior skipped.

### `tests/unit/test_ingestion.py`

```python
from pathlib import Path
import pytest

from ragworkshop.context import ApplicationContext
from ragworkshop.embeddings.fake import FakeEmbeddingProvider
from ragworkshop.operations import ingestion
from ragworkshop.operations.search import semantic_search
from ragworkshop.processors.chunks import TextChunker
from ragworkshop.storage.base import StoredObject


class MemoryStorage:
    def __init__(self):
        self.keys = set()
    def save(self, source, key, content_type):
        self.keys.add(key)
        return StoredObject(key, 1, content_type, "checksum")
    def delete(self, key):
        self.keys.remove(key)
        return True


class FailingRepository:
    def __init__(self):
        self.records = []
        self.deleted = []
    def ensure_space(self, model_name, dimensions):
        pass
    def add(self, documents):
        self.records = documents[:1]
        raise RuntimeError("partial database write")
    def delete(self, ids):
        self.deleted = ids
        self.records = [doc for doc in self.records if doc.id not in ids]


def test_partial_failure_removes_all_allocated_ids(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("python cat")
    storage = MemoryStorage()
    repo = FailingRepository()
    context = ApplicationContext(repo, storage, FakeEmbeddingProvider())
    with pytest.raises(RuntimeError, match="partial"):
        ingestion.ingest_document(path, context, TextChunker(5, 5))
    assert len(repo.deleted) == 2
    assert repo.records == []
    assert storage.keys == set()


def test_cleanup_failure_keeps_original_and_attempts_storage(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("python")
    storage = MemoryStorage()
    repo = FailingRepository()
    def fail(ids):
        raise OSError("cleanup unavailable")
    repo.delete = fail
    with pytest.raises(ExceptionGroup) as error:
        ingestion.ingest_document(path, ApplicationContext(repo, storage, FakeEmbeddingProvider()), TextChunker())
    assert isinstance(error.value.exceptions[0], RuntimeError)
    assert isinstance(error.value.exceptions[1], OSError)
    assert storage.keys == set()


def test_empty_file_compensates(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text(" ")
    storage = MemoryStorage()
    with pytest.raises(ValueError, match="no extractable"):
        ingestion.ingest_document(path, ApplicationContext(FailingRepository(), storage, FakeEmbeddingProvider()), TextChunker())
    assert storage.keys == set()


@pytest.mark.parametrize("query,limit", [(" ", 1), ("cat", 0), ("cat", True)])
def test_invalid_search(query, limit):
    with pytest.raises(ValueError):
        semantic_search(query, ApplicationContext(FailingRepository(), MemoryStorage(), FakeEmbeddingProvider()), limit)
```

### `tests/integration/test_database.py`

```python
import pytest
from ragworkshop.database.factory import get_database
from ragworkshop.database.models import Document
from ragworkshop.database.migrations import migrate


def test_persistence_filters_and_migrations(tmp_path):
    path = tmp_path / "persistent"
    backend = get_database("chroma", path)
    assert migrate(backend) == ["0001"]
    assert migrate(backend) == []
    repository = backend.documents()
    repository.ensure_space("test-space", 2)
    assert repository.search_by_vector([1.0, 0.0]) == []
    repository.add([Document("id-1", "source-1", "python", [1.0, 0.0], "documents/a.txt", {"page": 1}),
                    Document("id-2", "source-2", "cat", [0.0, 1.0], "documents/b.txt", {"page": 2})])
    fresh = get_database("chroma", path).documents()
    fresh.ensure_space("test-space", 2)
    result = fresh.search_by_vector([1.0, 0.0], filters={"source_id": "source-1", "page": 1})
    assert [match.document.id for match in result] == ["id-1"]
    assert result[0].distance == pytest.approx(0.0)
    assert result[0].document.metadata == {"page": 1}
    with pytest.raises(ValueError, match="space mismatch"):
        fresh.ensure_space("another-model", 2)
    with pytest.raises(ValueError, match="space mismatch"):
        fresh.ensure_space("test-space", 3)
    with pytest.raises(ValueError, match="dimension"):
        fresh.search_by_vector([1.0])
    fresh.delete(["id-1", "id-2"])
    assert fresh.search_by_vector([1.0, 0.0]) == []


def test_backend_factory_rejects_unknown(tmp_path):
    with pytest.raises(ValueError):
        get_database("postgres", tmp_path)
```

### `tests/e2e/test_pipeline.py`

```python
import json
import subprocess
from dataclasses import replace

from docx import Document as DocxDocument
from reportlab.pdfgen.canvas import Canvas
import pytest

from ragworkshop.config import load_settings
from ragworkshop.context import create_context
from ragworkshop.embeddings.fake import FakeEmbeddingProvider
from ragworkshop.operations.ingestion import ingest_document
from ragworkshop.operations.search import semantic_search
from ragworkshop.processors.chunks import TextChunker


@pytest.mark.parametrize("suffix", [".txt", ".docx", ".pdf"])
def test_deterministic_fake_provider_e2e(tmp_path, suffix):
    path = tmp_path / ("lesson" + suffix)
    if suffix == ".txt":
        path.write_text("python python modules", encoding="utf-8")
    elif suffix == ".docx":
        document = DocxDocument()
        document.add_paragraph("python python modules")
        document.save(path)
    else:
        canvas = Canvas(str(path), invariant=1)
        canvas.drawString(72, 720, "python python modules")
        canvas.save()
    settings = load_settings()
    assert settings.environment == "test"
    assert settings.database_path == tmp_path / "db"
    with create_context(settings, FakeEmbeddingProvider()) as context:
        indexed = ingest_document(path, context, TextChunker(100, 100))
        assert context.storage.exists(indexed[0].storage_key)
        destination = context.storage.get(indexed[0].storage_key, tmp_path / ("copy" + suffix))
        assert destination.read_bytes() == path.read_bytes()
    with create_context(settings, FakeEmbeddingProvider()) as fresh:
        results = semantic_search("python python", fresh)
        assert results[0].document.id == indexed[0].id
        assert results[0].document.metadata["filename"] == path.name
    other = replace(settings, environment="development", database_path=tmp_path / "isolated-dev-db",
                    storage_root=tmp_path / "isolated-dev-objects")
    with create_context(other, FakeEmbeddingProvider()) as isolated:
        assert semantic_search("python", isolated) == []
        assert isolated.storage.list() == []


def test_installed_cli_persists_across_processes(tmp_path):
    source = tmp_path / "lesson.txt"
    source.write_text("python modules", encoding="utf-8")
    def run(*args):
        result = subprocess.run(["rag-workshop", *map(str, args)], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    assert run("db:create") == {"database": "initialized"}
    assert run("db:migrate") == {"applied": ["0001"]}
    assert run("db:migrate") == {"applied": []}
    indexed = run("ingest", source)
    result = run("search", "python", "--source-id", indexed["source_id"])
    assert result[0]["document"]["storage_key"] == indexed["storage_key"]
    assert run("storage:list")[0]["key"] == indexed["storage_key"]
```

## Green and Refactor constraints

- Implement ChromaDocumentRepository against the supplied backend and record types.
- Validate the model identity and dimensions before indexing/search; use explicit vectors.
- Implement create_context with an optional injected provider and deterministic ownership rules.
- Implement ingest_document: store, extract/chunk, embed, persist, and compensate partial failures.
- Implement semantic_search with validation, same-space query embedding, filters, and source-bearing results.
- Use .env for the manual demo and monkeypatch temporary paths for automated tests.
- Verify migrations, fresh backend persistence, process restart persistence, environment isolation, and all three document formats.
- First satisfy one failing assertion, then rerun the current module tests.
- Refactor only after Green; preserve earlier commands and tests.
- Keep business behavior out of CLI branches. Use supplied interfaces and explicit arguments.
- Keep automated tests offline, with temporary paths for file/database interactions.
- No GPU, cloud credentials, or real GGUF is required.

## Input/output examples and expected observable behavior

**Working directory: `rag-workshop/`.** Output comments specify assertions, not byte-for-byte JSON formatting.

```bash
rag-workshop --env test db:create
# {"database": "initialized"}
rag-workshop --env test db:migrate
# {"applied": ["0001"]} on first migration; {"applied": []} thereafter
rag-workshop --env test ingest samples/sample.txt
# JSON contains indexed_chunks: 1, a generated source_id, and documents/<uuid>.txt storage_key
rag-workshop --env test search python --limit 3
# JSON array; each result has document.content, document.storage_key, source metadata, and distance
rag-workshop --env test storage:list
# original document key and checksum remain available
```

## Acceptance criteria

- Implement ChromaDocumentRepository against the supplied backend and record types.
- Validate the model identity and dimensions before indexing/search; use explicit vectors.
- Implement create_context with an optional injected provider and deterministic ownership rules.
- Implement ingest_document: store, extract/chunk, embed, persist, and compensate partial failures.
- Implement semantic_search with validation, same-space query embedding, filters, and source-bearing results.
- Use .env for the manual demo and monkeypatch temporary paths for automated tests.
- Verify migrations, fresh backend persistence, process restart persistence, environment isolation, and all three document formats.
- All provided current-module tests pass without changing the stated interfaces.
- Earlier module regression tests continue to pass.
- Explain one design-pattern participant and its tradeoff to a partner.

## Verification commands

**Working directory: your `rag-workshop/` project root; activate `.venv` first.**

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m pytest tests/e2e -q
python -m pytest -q
rag-workshop info
```

Validated cumulative result: **66 passed, 1 skipped, 1 warning**.

Expected: all implemented tests pass; the real-GGUF test is skipped unless explicitly opted in. This skip is expected, not evidence that real inference was tested.

## Optional stretch challenges

- Add a separately marked real-GGUF E2E test using a fresh database and the same provider for document/query embedding.
- Design a PostgreSQL adapter or authenticated FastAPI composition boundary without changing operations.
