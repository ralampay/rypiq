# Activity 5 solution: Database, configuration, and end-to-end integration

**Use this file during review or recovery.**

## Match to the exercise

Users need to ingest a document, restart the program, and retrieve relevant chunks with their source information. Compose the existing components with persistent Chroma and prove this works without a model download.

This solution uses exactly the activity signatures and requirements. It is a cumulative change: retain the previous module files and replace/add only the files listed here. The direct link below opens the complete cumulative checkpoint for this module.

[Complete Module 5 checkpoint](../../reference/module-05/)

## Red — completed tests first

Run these against the activity stubs and confirm the TODO failures before implementing. These are the same test contracts as Activity.md, with no changed expectations.

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

## Green — complete reference implementation

The exact filename precedes each complete file. Existing earlier-module files not shown here are unchanged. Empty package initializers and non-Python scaffolding are included so the checkpoint is reproducible.

ApplicationContext groups dependencies without hiding construction. create_context is a context manager that initializes migrations, enforces the embedding space, and respects provider ownership.

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
def create_context(settings: Settings, provider: EmbeddingProvider | None = None) -> Iterator[ApplicationContext]:
    owned = provider is None
    if provider is None:
        provider = get_embedding_provider(settings.embedding_provider, settings.embedding_model,
                                          settings.embedding_dimensions)
    try:
        backend = get_database(settings.database_adapter, settings.database_path)
        migrate(backend)
        repository = backend.documents()
        repository.ensure_space(provider.model_name, provider.dimensions)
        yield ApplicationContext(repository, get_storage(settings.storage_adapter, settings.storage_root), provider)
    finally:
        if owned and hasattr(provider, "close"):
            provider.close()
    # PersistentClient exposes no public close in the pinned Chroma version.
    # CLI subprocess exit releases its resources; never call private reset APIs.
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
from ragworkshop.context import create_context
from ragworkshop.database.factory import get_database
from ragworkshop.database.migrations import migrate
from ragworkshop.operations.ingestion import ingest_document
from ragworkshop.operations.search import semantic_search

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
    commands.add_parser("db:create")
    commands.add_parser("db:migrate")
    command = commands.add_parser("ingest")
    command.add_argument("path", type=Path)
    command.add_argument("--chunk-size", type=int, default=200)
    command.add_argument("--stride", type=int, default=150)
    command = commands.add_parser("search")
    command.add_argument("query")
    command.add_argument("--limit", type=int, default=5)
    command.add_argument("--source-id")
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
        elif args.command in ("db:create", "db:migrate"):
            database = get_database(settings.database_adapter, settings.database_path)
            if args.command == "db:create":
                database.create_database()
                result = {"database": "initialized"}
            else:
                result = {"applied": migrate(database)}
        else:
            with create_context(settings) as context:
                if args.command == "ingest":
                    documents = ingest_document(args.path, context, TextChunker(args.chunk_size, args.stride))
                    result = {"indexed_chunks": len(documents), "source_id": documents[0].source_id,
                              "storage_key": documents[0].storage_key}
                else:
                    filters = {"source_id": args.source_id} if args.source_id else None
                    result = [asdict(item) for item in semantic_search(args.query, context, args.limit, filters)]
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

Frozen dataclasses make the boundary records explicit and readable. Metadata is copied when mapping; frozen does not recursively freeze dictionaries.

### `src/ragworkshop/database/models.py`

```python
from dataclasses import dataclass
from ragworkshop.processors.chunks import Metadata


@dataclass(frozen=True)
class Document:
    id: str
    source_id: str
    content: str
    embedding: list[float]
    storage_key: str
    metadata: Metadata


@dataclass(frozen=True)
class SearchResult:
    document: Document
    distance: float
```

The Protocol specifies behavior without requiring inheritance. Implementations and lightweight fakes must preserve this contract. Validation belongs at boundaries so malformed provider output never reaches persistence.

### `src/ragworkshop/database/base.py`

```python
from typing import Protocol
from ragworkshop.database.models import Document, SearchResult
from ragworkshop.processors.chunks import Metadata


class DocumentRepository(Protocol):
    def ensure_space(self, model_name: str, dimensions: int) -> None: ...
    def add(self, documents: list[Document]) -> None: ...
    def delete(self, ids: list[str]) -> None: ...
    def search_by_vector(self, embedding: list[float], limit: int = 5,
                         filters: Metadata | None = None) -> list[SearchResult]: ...


class DatabaseBackend(Protocol):
    def create_database(self) -> None: ...
    def create_documents_collection(self) -> None: ...
    def applied_migrations(self) -> set[str]: ...
    def record_migration(self, version: str) -> None: ...
    def documents(self) -> DocumentRepository: ...
```

The backend owns client and collection lifecycle; the repository owns mapping and vector queries. Explicit embeddings prevent hidden inference. Collection metadata rejects mixing spaces.

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
        metadata = dict(self.collection.metadata or {})
        if "embedding_space" in metadata:
            if (metadata["embedding_space"] != model_name
                    or metadata["dimensions"] != dimensions):
                raise ValueError("Embedding space mismatch; use a new collection or rebuild")
        else:
            if self.collection.count():
                raise ValueError("Existing collection lacks embedding space metadata")
            self.collection.modify(metadata={**metadata, "embedding_space": model_name,
                                             "dimensions": dimensions})
        self.dimensions = dimensions

    def add(self, documents: list[Document]) -> None:
        if not documents:
            return
        if self.dimensions is None:
            raise ValueError("Call ensure_space before indexing")
        validate_vectors([document.embedding for document in documents], len(documents), self.dimensions)
        for document in documents:
            # Scalar metadata remains filterable; reserved fields cannot be overridden.
            metadata = {**document.metadata, "source_id": document.source_id,
                        "storage_key": document.storage_key}
            self.collection.add(ids=[document.id], documents=[document.content],
                                embeddings=[document.embedding], metadatas=[metadata])

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    def search_by_vector(self, embedding: list[float], limit: int = 5,
                         filters: Metadata | None = None) -> list[SearchResult]:
        if self.dimensions is None:
            raise ValueError("Call ensure_space before search")
        validate_vectors([embedding], 1, self.dimensions)
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        count = self.collection.count()
        if not count:
            return []
        where = None
        if filters:
            conditions = [{key: value} for key, value in filters.items()]
            where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
        result = self.collection.query(query_embeddings=[embedding], n_results=min(count, limit),
                                       where=where, include=["documents", "embeddings", "metadatas", "distances"])
        matches = []
        for i, record_id in enumerate(result["ids"][0]):
            metadata = dict(result["metadatas"][0][i])
            document = Document(record_id, metadata.pop("source_id"), result["documents"][0][i],
                                list(map(float, result["embeddings"][0][i])),
                                metadata.pop("storage_key"), metadata)
            matches.append(SearchResult(document, float(result["distances"][0][i])))
        return matches


class ChromaDatabaseBackend:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.path), settings=ChromaSettings(anonymized_telemetry=False))
        return self._client

    def create_database(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        _ = self.client

    def create_documents_collection(self) -> None:
        self.client.get_or_create_collection("documents", embedding_function=None,
                                            configuration={"hnsw": {"space": "cosine"}})

    def _migrations(self):
        return self.client.get_or_create_collection("ragworkshop_migrations", embedding_function=None)

    def applied_migrations(self) -> set[str]:
        return set(self._migrations().get(include=[])["ids"])

    def record_migration(self, version: str) -> None:
        self._migrations().upsert(ids=[version], embeddings=[[1.0]], documents=[version])

    def documents(self) -> ChromaDocumentRepository:
        return ChromaDocumentRepository(self.client.get_collection("documents", embedding_function=None))
```

A simple factory constructs a supported adapter using explicit arguments and rejects unknown names. It does not reach into global settings.

### `src/ragworkshop/database/factory.py`

```python
from pathlib import Path
from ragworkshop.database.base import DatabaseBackend


def get_database(adapter: str, path: Path) -> DatabaseBackend:
    if adapter != "chroma":
        raise ValueError(f"Unsupported database adapter: {adapter}")
    from ragworkshop.database.chroma import ChromaDatabaseBackend
    return ChromaDatabaseBackend(path)
```

Migration is a small command record containing a version and callable. Completion is recorded only after apply succeeds; the creation step is idempotent.

### `src/ragworkshop/database/migrations.py`

```python
from collections.abc import Callable
from dataclasses import dataclass
from ragworkshop.database.base import DatabaseBackend


@dataclass(frozen=True)
class Migration:
    version: str
    apply: Callable[[DatabaseBackend], None]


MIGRATIONS = [Migration("0001", lambda backend: backend.create_documents_collection())]


def migrate(backend: DatabaseBackend) -> list[str]:
    backend.create_database()
    applied = backend.applied_migrations()
    completed = []
    for migration in MIGRATIONS:
        if migration.version not in applied:
            migration.apply(backend)
            backend.record_migration(migration.version)
            completed.append(migration.version)
    return completed
```

The operation tracks owned side effects and allocated IDs. Compensation attempts both index and object cleanup while preserving the original exception.

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
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError("Input must be a regular file")
    context.documents.ensure_space(context.embeddings.model_name, context.embeddings.dimensions)
    source_id = str(uuid4())
    key = f"documents/{source_id}{source.suffix.lower()}"
    content_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
    stored = context.storage.save(source, key, content_type)
    ids: list[str] = []
    try:
        chunks = process_document(source, chunker)
        if not chunks:
            raise ValueError("Document has no extractable text; scanned PDF may require OCR")
        vectors = embed_chunks(chunks, context.embeddings)
        ids = [str(uuid4()) for chunk in chunks]
        documents = [Document(record_id, source_id, chunk.text, vector, stored.key,
                              {**chunk.metadata, "chunk_index": chunk.index, "start": chunk.start,
                               "end": chunk.end, "checksum": stored.checksum,
                               "embedding_space": context.embeddings.model_name})
                     for record_id, chunk, vector in zip(ids, chunks, vectors, strict=True)]
        context.documents.add(documents)
        return documents
    except Exception as original:
        # Include all allocated IDs: an adapter can fail after a partial write.
        failures = []
        try:
            context.documents.delete(ids)
        except Exception as cleanup:
            failures.append(cleanup)
        try:
            context.storage.delete(key)
        except Exception as cleanup:
            failures.append(cleanup)
        if failures:
            raise ExceptionGroup("Ingestion failed and cleanup was incomplete", [original, *failures]) from original
        raise
```

The operation validates the query, checks the embedding space, embeds once, and delegates ranking/filtering to the repository.

### `src/ragworkshop/operations/search.py`

```python
from ragworkshop.context import ApplicationContext
from ragworkshop.database.models import SearchResult
from ragworkshop.embeddings.base import validate_vectors
from ragworkshop.processors.chunks import Metadata


def semantic_search(query: str, context: ApplicationContext, limit: int = 5,
                    filters: Metadata | None = None) -> list[SearchResult]:
    if not query.strip():
        raise ValueError("Query must not be empty")
    if type(limit) is not int or limit < 1:
        raise ValueError("limit must be a positive integer")
    provider = context.embeddings
    context.documents.ensure_space(provider.model_name, provider.dimensions)
    vectors = provider.embed([query])
    validate_vectors(vectors, 1, provider.dimensions)
    return context.documents.search_by_vector(vectors[0], limit, filters)
```

## Refactor, pattern rationale, and alternatives

A repository with one add(list[Document]) method is sufficient for the use case; do not add unused generic CRUD or an ORM hierarchy. A transaction is preferable within a single relational database, but it does not automatically cover object storage. An outbox or staged-ingestion state machine could improve crash recovery later.

`DatabaseBackend` manages client initialization, collections, and migration bookkeeping. `DocumentRepository` manages application records and vector search. The Repository pattern hides Chroma's nested query dictionaries from operations; the backend Adapter hides connection and collection details. The Factory selects the backend. A dataclass `ApplicationContext` groups injected dependencies without service-location magic or global state.

`Document` maps one chunk to one ID, source ID, content string, embedding, object-storage key, and scalar metadata. `SearchResult` pairs it with a cosine distance: lower is closer, and it is not a calibrated probability. Original bytes remain in object storage. Source ID groups chunks from one ingest; each new ingest intentionally creates a new source ID rather than deduplicating by filename.

Chroma uses `PersistentClient`, explicit vectors, and `embedding_function=None` for both creation and reopening. No default embedding model should be downloaded. Collection-level metadata binds model identity and dimensions. `ensure_space` rejects both dimension mismatch and a different model with the same dimensions. This example supports one embedding space per collection and assumes a single CLI writer. Concurrent initialization, model upgrades, and reindexing require a more deliberate production migration strategy.

Migration `0001` idempotently creates the documents collection, then records completion in a separate collection using an explicit one-dimensional sentinel vector. A failed migration must not be marked complete. This is a minimal schema-lifecycle abstraction, not SQL Alembic or a distributed migration runner. `db:create` initializes persistence; `db:migrate` applies the collection migration; context creation also calls the idempotent migrator for predictable first use.

Configuration precedence is CLI arguments > existing environment > `.env` values loaded without overriding existing environment > selected YAML section > documented defaults. `RAG_WORKSHOP_ENV` selects development/test/production consistently. File-selection variables are `RAG_WORKSHOP_DATABASE_CONFIG` and `RAG_WORKSHOP_STORAGE_CONFIG`; path overrides are `RAG_WORKSHOP_DATABASE_PATH` and `RAG_WORKSHOP_STORAGE_ROOT`. Provider/model/dimensions have matching prefixed variables. YAML can include an `embedding` mapping under the database environment section. Unknown environments fail rather than quietly using development.

Secrets never belong in YAML or examples. Do not print Settings or environment dumps in logs. In this local-only implementation no credentials are required. `python-dotenv` is invoked at the boundary with `override=False`; pytest's `monkeypatch` selects temporary paths and an absent dotenv file. Tests that compare development settings do not initialize those paths. The E2E isolation test initializes a second environment only under its temporary directory.

Ingestion saves the original, extracts/chunks it, embeds in batches, allocates all record IDs, and persists them. Failures after saving compensate by deleting all allocated IDs (including potentially partially written records) and the newly saved object. Both cleanup steps are attempted independently. `ExceptionGroup` preserves the original failure when cleanup also fails. This is compensation, not an atomic transaction across two stores; a process crash may still leave orphans. Search validates the query, embeds it with the same provider, and delegates filtering and ranking to the repository.

The context manager closes only resources it owns; injected providers remain caller-owned. The pinned Chroma client has no public close method, so CLI process exit releases its resources. Do not teach private reset APIs as cleanup. Tests use temporary persistent paths and terminate CLI subprocesses.

A PostgreSQL/pgvector adapter would implement the same repository and backend contracts, translate metadata filters to parameterized queries, and supply its own migrations. A future FastAPI adapter would inject a context, require a secure header checked against an environment-provided secret (constant-time comparison), reject unauthenticated requests before operations, and restrict filesystem access. TLS, upload limits, authorization, and lifecycle management remain deployment requirements. No unauthenticated API is exposed by this course.

## Verification and expected test outcomes

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

**Working directory: `rag-workshop/`.**

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

## Edge cases and limitations

Empty queries, invalid limits, mismatched spaces, and invalid vectors fail before retrieval. Metadata filters use scalar equality and combine multiple entries with AND. Re-ingestion creates a new source, not an upsert. Cleanup failure preserves both errors; process crashes need an operational reconciliation mechanism beyond this workshop. A fresh backend in one process is not enough to prove restart persistence, so the CLI E2E uses separate processes.

## Solution architecture trace

Arrows show **the required deterministic E2E test across two context lifetimes**. The provider is reproducible, while parsers, object storage, and Chroma are real.

```text
context A: test settings + supplied FakeEmbeddingProvider
       |
       +--> initialize/migrate Chroma
       +--> ensure space = demo-keyword-v1, dimensions = 8
       +--> ingest fixture
              |
              +--> save original under key K
              +--> extract/chunk -> fake vectors
              +--> add records with IDs C and source_id S
       |
       v
leave context A (caller retains ownership of supplied provider)
       |
       v
context B: same test paths + fresh FakeEmbeddingProvider
       |
       +--> reopen collection; validate same embedding space
       +--> search "python python"
              |
              v
        SearchResult with stored ID C, key K, and source metadata

separate temporary environment paths -> empty index and empty object storage
installed CLI subprocess test        -> persistence across process exits
```

The repository handles Chroma's wire shape; operations handle use-case sequencing. Original bytes are verified through `storage.get`, not by reaching into the adapter root. The separate compensation tests inject a partial write and verify deletion of all allocated IDs plus the new object. Thus successful retrieval and safe failure handling have distinct executable evidence.

## Why the acceptance criteria are met

- Implement ChromaDocumentRepository against the supplied backend and record types. The completed code and assertions above make this behavior observable.
- Validate the model identity and dimensions before indexing/search; use explicit vectors. The completed code and assertions above make this behavior observable.
- Implement create_context with an optional injected provider and deterministic ownership rules. The completed code and assertions above make this behavior observable.
- Implement ingest_document: store, extract/chunk, embed, persist, and compensate partial failures. The completed code and assertions above make this behavior observable.
- Implement semantic_search with validation, same-space query embedding, filters, and source-bearing results. The completed code and assertions above make this behavior observable.
- Use .env for the manual demo and monkeypatch temporary paths for automated tests. The completed code and assertions above make this behavior observable.
- Verify migrations, fresh backend persistence, process restart persistence, environment isolation, and all three document formats. The completed code and assertions above make this behavior observable.

Tests are the executable acceptance evidence; successful imports alone are insufficient. Exact validation counts for all checkpoints are recorded in the workshop README. The optional real-model check is reported separately from default offline validation.
