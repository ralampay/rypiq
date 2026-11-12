# Introduction

Embed a user's question and retrieve the top-k matching protocol chunks from Chroma. Return the matching text and source references so the results can be inspected or supplied as context to an answer-generation model.

Complete [EmbeddingGeneration.md](EmbeddingGeneration.md), [DatabaseSetup.md](DatabaseSetup.md), and [StorageSetup.md](StorageSetup.md) first. This guide implements the **retrieval stage of RAG**. It does not generate an answer with an LLM.

# Objectives

- Test vector search and response mapping.
- Format and embed a user query with the same model used for ingestion.
- Return top-k chunks with source information and cosine distances.
- Test retrieval through the CLI using real Chroma and deterministic embeddings.

# Workflow

Use the same `protocol_embeddings` package and activated environment. Keep `chromadb==1.5.9` and `pytest` installed. For every section, write the tests first, run them to observe failure, then add the implementation.

Use one embedding model per database. A matching vector dimension alone does not make two models compatible. Keep the same GGUF file for ingestion and queries. To switch models, initialize a new database path and re-ingest the originals.

## Section 1

Add a search result model and vector-search repository.

### Tests to Write

Save as `tests/unit/test_vector_search.py`:

```python
from unittest.mock import Mock

import pytest

from protocol_embeddings.documents import Document, to_metadata
from protocol_embeddings.search import SearchRepository


def test_query_mapping_and_top_k():
    document = Document(content="Check calibration", embedding=[1.0, 0.0],
                        source="protocol.txt", storage_key="documents/source.txt")
    collection = Mock()
    collection.count.return_value = 2
    collection.query.return_value = {
        "ids": [[document.id]], "documents": [[document.content]],
        "embeddings": [[document.embedding]], "metadatas": [[to_metadata(document)]],
        "distances": [[0.1]],
    }
    result = SearchRepository(collection).search_by_vector([1.0, 0.0], top_k=5)
    collection.query.assert_called_once_with(
        query_embeddings=[[1.0, 0.0]], n_results=2,
        include=["documents", "metadatas", "embeddings", "distances"],
    )
    assert result[0].document == document
    assert result[0].distance == pytest.approx(0.1)


def test_empty_collection():
    collection = Mock()
    collection.count.return_value = 0
    assert SearchRepository(collection).search_by_vector([1.0, 0.0]) == []
    collection.query.assert_not_called()


@pytest.mark.parametrize("vector", [[], [float("nan")], [True], ["bad"]])
def test_invalid_query_vector(vector):
    with pytest.raises(ValueError):
        SearchRepository(Mock()).search_by_vector(vector)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k(top_k):
    with pytest.raises(ValueError):
        SearchRepository(Mock()).search_by_vector([1.0, 0.0], top_k)
```

Run `python -m pytest -q tests/unit/test_vector_search.py`.

### Implementation

Extend the document repository without changing its CRUD behavior. Chroma query results contain one nested result group per query vector. Here there is exactly one query, so read group zero. Use `query_embeddings` and `n_results` as described in the [Chroma query documentation](https://docs.trychroma.com/docs/querying-collections/query-and-get).

Save as `protocol_embeddings/search.py`:

```python
import math
from dataclasses import dataclass

from protocol_embeddings.documents import Document, from_record
from protocol_embeddings.repository import DocumentRepository


@dataclass(frozen=True)
class SearchResult:
    document: Document
    distance: float


def validate_top_k(top_k):
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top-k must be a positive integer")


def validate_vector(vector):
    if not vector or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) for value in vector
    ):
        raise ValueError("Expected a nonempty finite query vector")


class SearchRepository(DocumentRepository):
    def search_by_vector(self, embedding, top_k=5):
        validate_top_k(top_k)
        validate_vector(embedding)
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[embedding], n_results=min(top_k, count),
            include=["documents", "metadatas", "embeddings", "distances"],
        )
        return [SearchResult(
            document=from_record(record_id, result["documents"][0][i],
                                 result["embeddings"][0][i], result["metadatas"][0][i]),
            distance=float(result["distances"][0][i]),
        ) for i, record_id in enumerate(result["ids"][0])]
```

Top-k is an upper bound. Empty collections return an empty list; smaller collections return fewer than k results. Cosine distance is smaller for closer matches, as defined in [Chroma’s distance configuration](https://docs.trychroma.com/docs/collections/configure).

## Section 2

Validate the question, generate its embedding, and retrieve matching chunks.

### Tests to Write

Save as `tests/unit/test_semantic_query.py`:

```python
from unittest.mock import Mock

import pytest

from protocol_embeddings.search import semantic_search


def test_query_prefix_and_forwarding():
    provider = Mock()
    provider.embed.return_value = [[1.0, 0.0]]
    repository = Mock()
    repository.search_by_vector.return_value = ["match"]
    assert semantic_search("  How do I calibrate?  ", provider, repository, 3) == ["match"]
    provider.embed.assert_called_once_with([
        "task: search result | query: How do I calibrate?",
    ])
    repository.search_by_vector.assert_called_once_with([1.0, 0.0], top_k=3)


@pytest.mark.parametrize("query,top_k", [("", 3), ("  ", 3), (None, 3), ("question", 0)])
def test_invalid_arguments_skip_inference(query, top_k):
    provider = Mock()
    with pytest.raises(ValueError):
        semantic_search(query, provider, Mock(), top_k)
    provider.embed.assert_not_called()


@pytest.mark.parametrize("vectors", [[], [[]], [[1.0], [2.0]], [[float("inf")]]])
def test_invalid_model_response(vectors):
    provider = Mock()
    provider.embed.return_value = vectors
    repository = Mock()
    with pytest.raises(ValueError):
        semantic_search("question", provider, repository)
    repository.search_by_vector.assert_not_called()
```

Run `python -m pytest -q tests/unit/test_semantic_query.py`.

### Implementation

Replace `protocol_embeddings/search.py` with the complete version below. It preserves vector search and adds the query operation. The query prefix differs from the document prefix used during ingestion; both follow the [EmbeddingGemma prompt instructions](https://huggingface.co/google/embeddinggemma-300m).

Save as `protocol_embeddings/search.py`:

```python
import math
from dataclasses import dataclass

from protocol_embeddings.documents import Document, from_record
from protocol_embeddings.repository import DocumentRepository


@dataclass(frozen=True)
class SearchResult:
    document: Document
    distance: float


def validate_top_k(top_k):
    if type(top_k) is not int or top_k < 1:
        raise ValueError("top-k must be a positive integer")


def validate_vector(vector):
    if not vector or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) for value in vector
    ):
        raise ValueError("Expected a nonempty finite query vector")


class SearchRepository(DocumentRepository):
    def search_by_vector(self, embedding, top_k=5):
        validate_top_k(top_k)
        validate_vector(embedding)
        count = self.collection.count()
        if count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[embedding], n_results=min(top_k, count),
            include=["documents", "metadatas", "embeddings", "distances"],
        )
        return [SearchResult(
            document=from_record(record_id, result["documents"][0][i],
                                 result["embeddings"][0][i], result["metadatas"][0][i]),
            distance=float(result["distances"][0][i]),
        ) for i, record_id in enumerate(result["ids"][0])]


QUERY_PREFIX = "task: search result | query: "


def semantic_search(query, provider, repository, top_k=5):
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Query cannot be empty")
    validate_top_k(top_k)
    vectors = provider.embed([QUERY_PREFIX + query.strip()])
    if len(vectors) != 1:
        raise ValueError("Expected exactly one query embedding")
    validate_vector(vectors[0])
    return repository.search_by_vector(vectors[0], top_k=top_k)
```

## Section 3

Expose search through the CLI and test ranked retrieval.

### Tests to Write

Save as `tests/integration/test_query_commands.py`:

```python
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from protocol_embeddings.commands import main
from protocol_embeddings.database import Database, migrate
from protocol_embeddings.documents import Document
from protocol_embeddings.search import SearchRepository
from chromadb.errors import ChromaError


def test_ingest_then_query_real_database(tmp_path, monkeypatch, capsys):
    database = Database(tmp_path / "chroma")
    migrate(database)
    model = tmp_path / "fake.gguf"
    model.touch()
    engine = Mock()

    def embed(texts):
        # Controlled vectors test wiring and ranking, not linguistic model quality.
        return {"data": [{"index": i, "embedding":
                [1.0, 0.0] if "calibration" in text else [0.0, 1.0]}
                for i, text in enumerate(texts)]}

    engine.create_embedding.side_effect = embed
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=Mock(return_value=engine)))
    options = ["--db-path", str(database.path), "--storage-root", str(tmp_path / "files")]
    for filename, content in [("calibration.txt", "Check calibration."),
                              ("recording.txt", "Save the measurement.")]:
        source = tmp_path / filename
        source.write_text(content, encoding="utf-8")
        main(options + ["ingest", str(source), "--model", str(model)])
        capsys.readouterr()

    main(options + ["search", "What about calibration?", "--model", str(model), "--top-k", "5"])
    matches = json.loads(capsys.readouterr().out)
    assert len(matches) == 2
    assert matches[0]["filename"] == "calibration.txt"
    assert matches[0]["content"] == "Check calibration."
    assert matches[0]["distance"] < matches[1]["distance"]
    assert matches[0]["storage_key"].startswith("documents/")
    assert "embedding" not in matches[0]
    engine.create_embedding.assert_called_with([
        "task: search result | query: What about calibration?",
    ])

    # A new interpreter can query the persisted vectors without model inference.
    script = """
import sys
from protocol_embeddings.database import Database
from protocol_embeddings.search import SearchRepository
repository = SearchRepository(Database(sys.argv[1]).documents().collection)
assert repository.search_by_vector([1.0, 0.0], 1)[0].document.filename == 'calibration.txt'
"""
    subprocess.run([sys.executable, "-c", script, str(database.path)], check=True)


def test_empty_database_and_dimension_mismatch(tmp_path):
    database = Database(tmp_path / "chroma")
    migrate(database)
    repository = SearchRepository(database.documents().collection)
    assert repository.search_by_vector([1.0, 0.0]) == []
    repository.create(Document(content="Step", embedding=[1.0, 0.0]))
    with pytest.raises(ChromaError, match="dimension"):
        repository.search_by_vector([1.0, 0.0, 0.0])


def test_cli_reports_dimension_mismatch(tmp_path, monkeypatch, capsys):
    database = Database(tmp_path / "chroma")
    migrate(database)
    database.documents().create(Document(content="Step", embedding=[1.0, 0.0]))
    provider = Mock()
    provider.embed.return_value = [[1.0, 0.0, 0.0]]
    monkeypatch.setattr("protocol_embeddings.commands.LocalEmbeddingProvider",
                        Mock(return_value=provider))
    with pytest.raises(SystemExit) as error:
        main(["--db-path", str(database.path), "search", "question",
              "--model", "fake.gguf"])
    assert error.value.code == 2
    assert "dimension" in capsys.readouterr().err


def test_cli_reports_missing_model(tmp_path, capsys):
    database = Database(tmp_path / "chroma")
    migrate(database)
    with pytest.raises(SystemExit) as error:
        main(["--db-path", str(database.path), "search", "question",
              "--model", str(tmp_path / "missing.gguf")])
    assert error.value.code == 2
    assert "Embedding model not found" in capsys.readouterr().err


@pytest.mark.parametrize("query,top_k", [(" ", "1"), ("question", "0")])
def test_cli_rejects_invalid_query(tmp_path, capsys, query, top_k):
    with pytest.raises(SystemExit) as error:
        main(["--db-path", str(tmp_path / "chroma"), "search", query,
              "--model", "missing.gguf", "--top-k", top_k])
    assert error.value.code == 2
    assert "positive top-k" in capsys.readouterr().err
```

Run `python -m pytest -q tests/integration/test_query_commands.py`. These tests use real Chroma collections and the full ingestion/query path, substituting only the inference engine.

### Implementation

Replace the command module with this complete version. It retains all database, storage, ingestion, and listing commands.

Save as `protocol_embeddings/commands.py`:

```python
import argparse
import json

from chromadb.errors import ChromaError

from protocol_embeddings.database import Database, migrate
from protocol_embeddings.storage import LocalStorage, save_file
from protocol_embeddings.ingestion import ingest_document
from protocol_embeddings.local import LocalEmbeddingProvider
from protocol_embeddings.search import SearchRepository, semantic_search


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
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--model", required=True)
    search.add_argument("--top-k", type=int, default=5)
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
        elif args.command == "search":
            # Validate arguments before loading model weights.
            if not args.query.strip() or args.top_k < 1:
                raise ValueError("Provide a nonblank query and positive top-k")
            repository = SearchRepository(database.documents().collection)
            provider = LocalEmbeddingProvider(args.model)
            matches = semantic_search(args.query, provider, repository, args.top_k)
            result = [{**match.document.summary(), "distance": match.distance}
                      for match in matches]
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    except (OSError, ValueError, RuntimeError, ChromaError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
```

Run every tutorial test:

```bash
python -m pytest -q tests
```

For an optional real-model smoke test, use the model file supplied for embedding generation. Use a fresh database path so deterministic test vectors and real embeddings never mix:

```bash
python -m pip install llama-cpp-python
python -m protocol_embeddings.commands --db-path data/query-demo db:create
python -m protocol_embeddings.commands --db-path data/query-demo db:migrate
cat > calibration.txt <<'TEXT'
Before measuring a sample, check instrument calibration against the reference standard.
Record the calibration result and the time of the check.
TEXT

python -m protocol_embeddings.commands --db-path data/query-demo --storage-root data/query-files \
  ingest calibration.txt --model models/embeddinggemma-300m-Q4_0.gguf
python -m protocol_embeddings.commands --db-path data/query-demo \
  search 'What should I check before measuring a sample?' \
  --model models/embeddinggemma-300m-Q4_0.gguf --top-k 3
```

The JSON output contains up to three matching chunks with `content`, `source`, `source_id`, `storage_key`, `chunk_index`, offsets, and `distance`. It contains retrieved evidence rather than a generated answer. A relevant chunk's presence is useful for manual inspection; do not assert an exact floating-point distance for real model output.

If a query exceeds the embedding model's token context, shorten it. If Chroma reports incompatible vector dimensions, confirm that ingestion and querying use the same GGUF file and database. Equal dimensions do not detect a different model: this tutorial relies on using one model per database, not automatic model-identity enforcement.

