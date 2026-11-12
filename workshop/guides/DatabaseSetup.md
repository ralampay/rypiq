# Introduction

Build a persistent Chroma database for embedded protocol chunks. Define a document model, store and retrieve records through a repository, and initialize collections with repeatable migrations.

# Objectives

- Write model and repository unit tests before implementation.
- Store explicit embeddings and source metadata in Chroma.
- Configure Chroma with `config/database.yml` and run `rypiq db:create` and `rypiq db:migrate`.
- Test persistence and migration behavior against a temporary database.

# Workflow

Activate your tutorial environment and run commands from the Rypiq project root, with your `protocol_embeddings` tutorial package available there. Install the project to register the `rypiq` command and the dependencies used by these examples:

```bash
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pip install 'chromadb==1.5.9'
mkdir -p tests/unit tests/integration
```

For Sections 1 and 2, save the tests, run the stated command to observe a failure, add the implementation, then rerun. Section 3 uses the existing project database implementation. All code blocks labelled with a filename contain that file's complete contents.

Use the installed `rypiq` entry point for database commands. Database paths and collection names come from `config/database.yml`. Keep the embedding-only `protocol_embeddings.cli` module unchanged.

## Section 1

Define the document model and its Chroma metadata representation.

### Tests to Write

Save as `tests/unit/test_documents.py`:

```python
from protocol_embeddings.documents import Document, from_record, to_metadata


def test_document_defaults_are_independent():
    first = Document(content="First", embedding=[1.0, 0.0])
    second = Document(content="Second", embedding=[0.0, 1.0])
    assert first.id != second.id
    assert first.created_at


def test_metadata_round_trip():
    document = Document(
        content="Check calibration", embedding=[1.0, 0.0],
        source_id="protocol-1", storage_key="documents/source.txt",
        filename="source.txt", source="source.txt", chunk_index=2,
        start=10, end=27, embedding_model="model.gguf",
    )
    metadata = to_metadata(document)
    assert "embedding" not in metadata
    assert "content" not in metadata
    restored = from_record(document.id, document.content, document.embedding, metadata)
    assert restored == document
    assert "embedding" not in document.summary()
```

Run `python -m pytest -q tests/unit/test_documents.py`.

### Implementation

Use scalar metadata values that Chroma can store directly. Keep the vector and text in their dedicated Chroma fields.

Save as `protocol_embeddings/documents.py`:

```python
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from uuid import uuid4


@dataclass
class Document:
    content: str
    embedding: list[float]
    id: str = field(default_factory=lambda: str(uuid4()))
    source_id: str = ""
    storage_key: str = ""
    filename: str = ""
    source: str = ""
    chunk_index: int = 0
    start: int = 0
    end: int = 0
    embedding_model: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def summary(self):
        values = asdict(self)
        values.pop("embedding")
        return values


def to_metadata(document):
    values = document.summary()
    values.pop("id")
    values.pop("content")
    return values


def from_record(record_id, content, embedding, metadata):
    return Document(id=record_id, content=content,
                    embedding=[float(value) for value in embedding], **metadata)
```

## Section 2

Build a repository that maps document records to Chroma calls.

### Tests to Write

Save as `tests/unit/test_repository.py`:

```python
from unittest.mock import Mock

import pytest

from protocol_embeddings.documents import Document, to_metadata
from protocol_embeddings.repository import DocumentRepository


def test_create_passes_explicit_vectors():
    collection = Mock()
    document = Document(content="Step one", embedding=[1.0, 0.0])
    assert DocumentRepository(collection).create(document) == document
    collection.add.assert_called_once_with(
        ids=[document.id], documents=[document.content],
        embeddings=[document.embedding], metadatas=[to_metadata(document)],
    )


def test_get_and_list_deserialize():
    document = Document(content="Step one", embedding=[1.0, 0.0])
    collection = Mock()
    collection.get.return_value = {
        "ids": [document.id], "documents": [document.content],
        "embeddings": [document.embedding], "metadatas": [to_metadata(document)],
    }
    repository = DocumentRepository(collection)
    assert repository.get(document.id) == document
    assert repository.list(limit=2, offset=1) == [document]
    assert collection.get.call_args.kwargs["offset"] == 1
    collection.get.return_value = {"ids": []}
    assert repository.get("absent") is None
    assert repository.delete("absent") is False
    collection.delete.assert_not_called()


@pytest.mark.parametrize("limit,offset", [(0, 0), (2, -1), (True, 0)])
def test_invalid_pagination(limit, offset):
    with pytest.raises(ValueError):
        DocumentRepository(Mock()).list(limit, offset)
```

Run `python -m pytest -q tests/unit/test_repository.py`.

### Implementation

The repository accepts a Chroma collection, making unit tests independent of disk storage. Embeddings are supplied explicitly; Chroma will not generate them.

Save as `protocol_embeddings/repository.py`:

```python
from protocol_embeddings.documents import from_record, to_metadata


class DocumentRepository:
    def __init__(self, collection):
        self.collection = collection

    def create(self, document):
        self.collection.add(
            ids=[document.id], documents=[document.content],
            embeddings=[document.embedding], metadatas=[to_metadata(document)],
        )
        return document

    def _records(self, result):
        return [from_record(record_id, result["documents"][i],
                            result["embeddings"][i], result["metadatas"][i])
                for i, record_id in enumerate(result["ids"])]

    def get(self, record_id):
        result = self.collection.get(
            ids=[record_id], include=["documents", "embeddings", "metadatas"],
        )
        records = self._records(result)
        return records[0] if records else None

    def delete(self, record_id):
        if self.get(record_id) is None:
            return False
        self.collection.delete(ids=[record_id])
        return True

    def list(self, limit=100, offset=0):
        if (type(limit) is not int or type(offset) is not int
                or limit < 1 or offset < 0):
            raise ValueError("Invalid pagination")
        return self._records(self.collection.get(
            limit=limit, offset=offset,
            include=["documents", "embeddings", "metadatas"],
        ))
```

## Section 3

Configure and initialize the database using the project's existing CLI. The model and repository in Sections 1 and 2 illustrate the persistence mapping; the actual commands use `app.database` and `app.models.document`.

### Configuration

Save as `config/database.yml`:

```yaml
default: &default
  adapter: chromadb
  collections:
    documents: documents

development:
  <<: *default
  path: storage/chroma/development

test:
  <<: *default
  path: storage/chroma/test

production:
  <<: *default
  path: storage/chroma/production
```

The project loads this file through `app.database.config.load_database_config`. `DATABASE_ENV` selects the environment and defaults to `development`. `RYPIQ_DATABASE_CONFIG` can point to an alternative YAML file. Relative database paths are resolved against the configuration file's grandparent directory (the project root for `config/database.yml`), not the current working directory. Use `adapter: chromadb`, as expected by the project's database factory.

### Tests to Write

Save as `tests/integration/test_database.py`:

```python
import os
import subprocess

import pytest
import yaml

from app.database.factory import get_database
from app.database.migrations import migrate
from app.models.document import Document


@pytest.fixture
def database_config(tmp_path, monkeypatch):
    config_file = tmp_path / "config" / "database.yml"
    config_file.parent.mkdir()
    config_file.write_text(yaml.safe_dump({
        "test": {
            "adapter": "chromadb",
            "path": "storage/chroma/test",
            "collections": {"documents": "documents"},
        },
    }), encoding="utf-8")
    monkeypatch.setenv("RYPIQ_DATABASE_CONFIG", str(config_file))
    monkeypatch.setenv("DATABASE_ENV", "test")
    return config_file


def test_persistence_and_delete(database_config):
    database = get_database()
    assert database.path == database_config.parent.parent / "storage/chroma/test"
    database.create_database()
    assert migrate(database) == ["0001"]
    assert migrate(database) == []
    repository = database.documents()
    document = Document(content="Check calibration", embedding=[1.0, 0.0])
    repository.create(document)
    reopened = get_database().documents()
    assert reopened.get(document.id) == document
    assert reopened.list() == [document]
    assert reopened.delete(document.id) is True
    assert reopened.list() == []


def test_cli_repeated_initialization(database_config):
    def run(command):
        return subprocess.run(
            ["rypiq", command], env=os.environ.copy(),
            check=True, capture_output=True, text=True,
        ).stdout.strip()

    for _ in range(2):
        assert run("db:create") == "Database initialized"
    assert run("db:migrate") == "Applied migration: 0001"
    assert run("db:migrate") == "No pending migrations"
```

These tests supply a temporary `database.yml` to both the factory and the CLI subprocesses, keeping test data separate from your working database.

Run the tutorial checks:

```bash
python -m pytest -q tests/unit/test_documents.py tests/unit/test_repository.py tests/integration/test_database.py
```

### Initialize the Database

With the default development configuration, run:

```bash
rypiq db:create
rypiq db:migrate
```

`rypiq db:create` initializes persistent storage and prints `Database initialized`. `rypiq db:migrate` creates the configured documents collection with cosine distance and records migration `0001` in `rypiq_schema_migrations`. On a fresh database it prints `Applied migration: 0001`. Running it again prints `No pending migrations`.

To select the test environment explicitly:

```bash
DATABASE_ENV=test rypiq db:create
DATABASE_ENV=test rypiq db:migrate
```

The implementation lives in `app/database/config.py`, `app/database/factory.py`, `app/database/backends/chroma.py`, and `app/database/migrations.py`; `app/cli.py` dispatches the commands. Application code uses `get_database().documents()` to access the configured repository after migration.

Migration actions must be safe to retry if application succeeds but recording the version fails. Run migrations from one process at a time.

Use a single embedding model and dimension per documents collection. To change models, select a new database path in `config/database.yml`, migrate it, and re-ingest the source files. Continue with [StorageSetup.md](StorageSetup.md).

### Implementations

#### `app/models/search.py`

```
from dataclasses import dataclass

from app.models.document import Document


@dataclass(frozen=True)
class SearchResult:
    document: Document
    distance: float
```

#### `app/database/repositories/chroma.py`

```
from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID

from app.database.repositories.base import (
    DocumentRepository,
)
from app.models.document import Document
from app.models.search import SearchResult


class ChromaDocumentRepository(DocumentRepository):

    def __init__(self, collection):
        self.collection = collection

    def _metadata(self, document: Document) -> dict:
        return {
            "storage_key": document.storage_key or "",
            "filename": document.filename,
            "content_type": document.content_type,
            "source": document.source,
            "source_id": (
                str(document.source_id)
                if document.source_id else ""
            ),
            "chunk_index": document.chunk_index,
            "embedding_model": document.embedding_model,
            "metadata_json": json.dumps(
                document.metadata,
                allow_nan=False,
            ),
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
        }

    def _deserialize(self, result: dict, i: int) -> Document:
        metadata = result["metadatas"][i]
        source_id = metadata.get("source_id", "")
        embedding = result["embeddings"][i]

        return Document(
            id=UUID(result["ids"][i]),
            content=result["documents"][i],
            embedding=embedding.tolist(),
            storage_key=(
                metadata.get("storage_key") or None
            ),
            filename=metadata.get("filename", ""),
            content_type=metadata.get("content_type", ""),
            source=metadata.get("source", ""),
            source_id=UUID(source_id) if source_id else None,
            chunk_index=metadata["chunk_index"],
            embedding_model=metadata["embedding_model"],
            metadata=json.loads(
                metadata.get("metadata_json", "{}")
            ),
            created_at=datetime.fromisoformat(
                metadata["created_at"]
            ),
            updated_at=datetime.fromisoformat(
                metadata["updated_at"]
            ),
        )

    def create(self, document: Document) -> Document:
        self.collection.add(
            ids=[str(document.id)],
            documents=[document.content],
            embeddings=[document.embedding],
            metadatas=[self._metadata(document)],
        )
        return document

    def get(self, id: UUID) -> Document | None:
        result = self.collection.get(
            ids=[str(id)],
            include=[
                "documents",
                "embeddings",
                "metadatas",
            ],
        )

        if not result["ids"]:
            return None

        return self._deserialize(result, 0)

    def update(self, document: Document) -> Document:
        if self.get(document.id) is None:
            raise KeyError(document.id)

        document.updated_at = datetime.now(timezone.utc)

        self.collection.update(
            ids=[str(document.id)],
            documents=[document.content],
            embeddings=[document.embedding],
            metadatas=[self._metadata(document)],
        )
        return document

    def delete(self, id: UUID) -> bool:
        if self.get(id) is None:
            return False

        self.collection.delete(ids=[str(id)])
        return True

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        if limit < 1 or offset < 0:
            raise ValueError("Invalid pagination")

        result = self.collection.get(
            limit=limit,
            offset=offset,
            include=[
                "documents",
                "embeddings",
                "metadatas",
            ],
        )

        return [
            self._deserialize(result, i)
            for i in range(len(result["ids"]))
        ]

    def search_by_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:

        if not embedding:
            raise ValueError("Embedding cannot be empty")

        if limit < 1:
            raise ValueError("Limit must be positive")

        kwargs = {
            "query_embeddings": [embedding],
            "n_results": limit,
        }

        if filters is not None:
            kwargs["where"] = filters

        result = self.collection.query(
            **kwargs,
            include=[
                "documents",
                "metadatas",
                "embeddings",
                "distances",
            ],
        )

        matches = []

        for index, record_id in enumerate(result["ids"][0]):

            document_data = {
                "ids": [record_id],
                "documents": [
                    result["documents"][0][index]
                ],
                "metadatas": [
                    result["metadatas"][0][index]
                ],
                "embeddings": [
                    result["embeddings"][0][index]
                ],
            }

            document = self._deserialize(
                document_data,
                0,
            )

            matches.append(
                SearchResult(
                    document=document,
                    distance=float(
                        result["distances"][0][index]
                    ),
                )
            )

        return matches
```

#### `app/database/chroma.py`

```
from pathlib import Path

import chromadb

from app.database.base import DatabaseBackend
from app.database.repositories.chroma import (
    ChromaDocumentRepository,
)


class ChromaBackend(DatabaseBackend):

    MIGRATIONS_COLLECTION = "rypiq_schema_migrations"

    def __init__(self, config: dict):
        self.path = Path(config["path"])
        self.collection_name = config.get(
            "collections", {}
        ).get("documents", "documents")

        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.path)
            )

        return self._client

    def create_database(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        _ = self.client

    def create_documents_collection(self) -> None:
        self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
        )

    def _migrations(self):
        return self.client.get_or_create_collection(
            name=self.MIGRATIONS_COLLECTION,
            embedding_function=None,
        )

    def applied_migrations(self) -> set[str]:
        result = self._migrations().get(
            include=[],
        )
        return set(result["ids"])

    def record_migration(
        self,
        version: str,
        description: str,
    ) -> None:
        self._migrations().add(
            ids=[version],
            documents=[description],
            embeddings=[[1.0]],
        )

    def documents(self) -> ChromaDocumentRepository:
        collection = self.client.get_collection(
            name=self.collection_name,
            embedding_function=None,
        )

        return ChromaDocumentRepository(collection)
```

#### `app/database/config.py`

```
import os
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_database_config() -> dict:
    config_file = Path(
        os.getenv(
            "RYPIQ_DATABASE_CONFIG",
            PROJECT_ROOT / "config/database.yml",
        )
    ).expanduser().resolve()

    environment = os.getenv(
        "DATABASE_ENV",
        "development",
    )

    with config_file.open(encoding="utf-8") as file:
        settings = yaml.safe_load(file)

    if not isinstance(settings, dict):
        raise ValueError("Invalid database configuration")

    if environment not in settings:
        raise ValueError(
            f"Unknown database environment: {environment}"
        )

    config = dict(settings[environment])

    if config.get("adapter") == "chromadb":
        path = Path(config["path"]).expanduser()

        if not path.is_absolute():
            path = config_file.parent.parent / path

        config["path"] = str(path.resolve())

    return config
```

#### `app/database/factory.py`

```
from app.database.base import DatabaseBackend
from app.database.config import load_database_config


def get_database(
    config: dict | None = None,
) -> DatabaseBackend:

    settings = (
        config
        if config is not None
        else load_database_config()
    )

    adapter = settings.get("adapter")

    if adapter == "chromadb":
        from app.database.backends.chroma import ChromaBackend

        return ChromaBackend(settings)

    raise ValueError(
        f"Unsupported database adapter: {adapter}"
    )
```

#### `app/database/base.py`

```
from abc import ABC, abstractmethod

from app.database.repositories.base import (
    DocumentRepository,
)


class DatabaseBackend(ABC):

    @abstractmethod
    def create_database(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def create_documents_collection(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def applied_migrations(self) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def record_migration(
        self,
        version: str,
        description: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def documents(self) -> DocumentRepository:
        raise NotImplementedError
```

#### `app/database/migrations.py`

```
from dataclasses import dataclass
from typing import Callable

from app.database.base import DatabaseBackend


@dataclass(frozen=True)
class Migration:
    version: str
    description: str
    apply: Callable[[DatabaseBackend], None]


MIGRATIONS = [
    Migration(
        version="0001",
        description="Create documents collection",
        apply=lambda db: db.create_documents_collection(),
    ),
]


def migrate(database: DatabaseBackend) -> list[str]:
    applied = database.applied_migrations()
    completed = []

    for migration in sorted(
        MIGRATIONS,
        key=lambda item: item.version,
    ):
        if migration.version in applied:
            continue

        migration.apply(database)

        database.record_migration(
            migration.version,
            migration.description,
        )

        completed.append(migration.version)

    return completed
```
