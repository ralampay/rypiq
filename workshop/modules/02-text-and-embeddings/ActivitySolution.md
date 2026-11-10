# Activity 2 solution: Text handling and embedding abstraction

**Use this file during review or recovery.**

## Match to the exercise

Incoming content can be plain text, Markdown source, or JSON with a configurable text field. Add interchangeable adapters so the CLI and embedding pipeline do not accumulate format-specific conditionals.

This solution uses exactly the activity signatures and requirements. It is a cumulative change: retain the previous module files and replace/add only the files listed here. The direct link below opens the complete cumulative checkpoint for this module.

[Complete Module 2 checkpoint](../../reference/module-02/)

## Red — completed tests first

Run these against the activity stubs and confirm the TODO failures before implementing. These are the same test contracts as Activity.md, with no changed expectations.

### `tests/unit/test_text.py`

```python
import pytest

from ragworkshop.processors.chunks import TextChunker
from ragworkshop.processors.formats import PROCESSORS, get_processor
from ragworkshop.embeddings.fake import FakeEmbeddingProvider
from ragworkshop.embeddings.factory import get_embedding_provider
from ragworkshop.embeddings.base import validate_vectors
from ragworkshop.operations.embedding import embed_chunks


def test_boundaries():
    chunks = TextChunker(4, 3).chunk("ABCDEFGHIJ", {"source": "memo"})
    assert [(c.start, c.end, c.text) for c in chunks] == [
        (0, 4, "ABCD"), (3, 7, "DEFG"), (6, 10, "GHIJ")]
    assert [c.index for c in chunks] == [0, 1, 2]
    assert all(c.metadata == {"source": "memo"} for c in chunks)


@pytest.mark.parametrize("size,stride", [(0, 1), (4, 0), (4, 5), (-1, 1), (True, 1), (4, 1.5)])
def test_bad_parameters(size, stride):
    with pytest.raises(ValueError):
        TextChunker(size, stride)


@pytest.mark.parametrize("text,expected", [("", []), ("ab", ["ab"]),
    ("aaaaaa", ["aaaa", "aaa"]), ("猫🐍éab", ["猫🐍éa", "ab"])])
def test_edges(text, expected):
    assert [c.text for c in TextChunker(4, 3).chunk(text)] == expected


def test_no_overlap():
    assert [c.text for c in TextChunker(3, 3).chunk("abcdefg")] == ["abc", "def", "g"]


@pytest.mark.parametrize("name,text,field,expected", [
    ("plain", "python", "text", "python"),
    ("markdown", "# python", "text", "# python"),
    ("json", '{"body": "python"}', "body", "python")])
def test_formats_and_pipeline(name, text, field, expected):
    processor = get_processor(name, TextChunker(30, 30), field)
    chunks = processor.process(text, {"source": "memo"})
    assert chunks[0].text == expected
    assert chunks[0].metadata["source"] == "memo"
    vectors = embed_chunks(chunks, FakeEmbeddingProvider())
    assert vectors == [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]


@pytest.mark.parametrize("text", ["{", "[]", "{}", '{"text": 4}'])
def test_bad_json(text):
    with pytest.raises(ValueError):
        get_processor("json", TextChunker()).process(text)


def test_registry_extensible(monkeypatch):
    builder = PROCESSORS["plain"]
    monkeypatch.setitem(PROCESSORS, "custom", builder)
    assert get_processor("custom", TextChunker()).process("hi")[0].text == "hi"
    with pytest.raises(ValueError):
        get_processor("unknown", TextChunker())


def test_factory_and_fake():
    provider = get_embedding_provider()
    assert provider.embed([]) == []
    assert provider.embed(["cat", "python"]) == provider.embed(["cat", "python"])
    assert provider.embed(["cat", "python"])[0] != provider.embed(["cat", "python"])[1]
    with pytest.raises(ValueError):
        get_embedding_provider("unknown")
    with pytest.raises(ValueError):
        get_embedding_provider("local")


def test_batch_order_and_error_propagation():
    class RecordingProvider:
        model_name = "recording"
        dimensions = 1
        def __init__(self):
            self.calls = []
        def embed(self, texts):
            self.calls.append(texts)
            return [[float(ord(text[0]))] for text in texts]
    provider = RecordingProvider()
    chunks = TextChunker(1, 1).chunk("abc")
    assert embed_chunks(chunks, provider, 2) == [[97.0], [98.0], [99.0]]
    assert provider.calls == [["a", "b"], ["c"]]
    def fail(texts):
        raise RuntimeError("provider unavailable")
    provider.embed = fail
    with pytest.raises(RuntimeError, match="unavailable"):
        embed_chunks(chunks, provider)
    with pytest.raises(ValueError):
        embed_chunks(chunks, provider, 0)


@pytest.mark.parametrize("vectors,count,dimensions", [([], 1, 2), ([[1]], 1, 2),
    ([[float("nan")]], 1, 1), ([[True]], 1, 1), ([[0.0]], 1, 1)])
def test_invalid_vectors(vectors, count, dimensions):
    with pytest.raises(ValueError):
        validate_vectors(vectors, count, dimensions)
```

### `tests/unit/test_local.py`

```python
import sys
from types import SimpleNamespace

import pytest

from ragworkshop.embeddings.factory import get_embedding_provider


def test_local_without_model_runtime(monkeypatch, tmp_path):
    path = tmp_path / "placeholder.gguf"
    path.write_bytes(b"test fixture, not a model")
    class Model:
        def __init__(self, **kwargs):
            assert kwargs["embedding"] is True
        def create_embedding(self, texts):
            return {"data": [{"index": i, "embedding": [float(i + 1), 1.0]}
                             for i in reversed(range(len(texts)))]}
        def close(self):
            pass
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=Model))
    provider = get_embedding_provider("local", str(path), 2)
    assert provider.embed(["a", "b"]) == [[1.0, 1.0], [2.0, 1.0]]
    assert provider.embed([]) == []
    provider.close()
```

### `tests/integration/test_real_model.py`

```python
import os

import pytest

from ragworkshop.embeddings.factory import get_embedding_provider


@pytest.mark.real_model
def test_real_gguf():
    # Dedicated test variables survive the application-settings isolation fixture.
    path = os.getenv("RAG_WORKSHOP_TEST_MODEL")
    if not path:
        pytest.fail("Set RAG_WORKSHOP_TEST_MODEL when opting into real model tests")
    dimensions = int(os.environ["RAG_WORKSHOP_TEST_DIMENSIONS"])
    provider = get_embedding_provider("local", path, dimensions)
    try:
        first = provider.embed(["Python modules", "Vector retrieval"])
        assert len(first) == 2
        assert len(first[0]) == dimensions
        assert provider.embed(["Python modules"])[0] == pytest.approx(first[0], abs=1e-5)
    finally:
        provider.close()
```

### `tests/integration/test_text_cli.py`

```python
import json
import subprocess


def test_process_and_embed_legacy_flags():
    processed = subprocess.run(["rag-workshop", "process", "ABCDEFGHIJ", "--chunk-size", "4", "--stride", "3"],
                               text=True, capture_output=True)
    assert processed.returncode == 0, processed.stderr
    assert [item["text"] for item in json.loads(processed.stdout)["chunks"]] == ["ABCD", "DEFG", "GHIJ"]
    embedded = subprocess.run(["rag-workshop", "embed", "python", "--provider", "fake", "--dimensions", "8"],
                              text=True, capture_output=True)
    assert embedded.returncode == 0, embedded.stderr
    assert json.loads(embedded.stdout)["vectors"] == [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
```

## Green — complete reference implementation

The exact filename precedes each complete file. Existing earlier-module files not shown here are unchanged. Empty package initializers and non-Python scaffolding are included so the checkpoint is reproducible.

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

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-workshop")
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
    args = parser.parse_args(argv)
    provider = None
    try:
        if args.command == "info":
            print("ragworkshop: modular ingestion and retrieval")
            return 0
        if args.command in ("process", "embed"):
            processor = get_processor(args.format, TextChunker(args.chunk_size, args.stride), args.field)
            chunks = processor.process(args.text)
            result = {"chunks": [asdict(chunk) for chunk in chunks]}
            if args.command == "embed":
                provider = get_embedding_provider(args.provider or "fake", args.model or "", args.dimensions if args.dimensions is not None else 8)
                result["vectors"] = embed_chunks(chunks, provider)
                result["model"] = provider.model_name
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

TextChunk stores offsets with content and metadata. TextChunker owns validation and window boundaries, copies metadata per chunk, and avoids a redundant final overlapping tail.

### `src/ragworkshop/processors/chunks.py`

```python
from dataclasses import dataclass, field

Metadata = dict[str, str | int | float | bool]


@dataclass(frozen=True)
class TextChunk:
    index: int
    start: int
    end: int
    text: str
    metadata: Metadata = field(default_factory=dict)


class TextChunker:
    def __init__(self, chunk_size: int = 200, stride: int = 150):
        if (type(chunk_size) is not int or type(stride) is not int
                or not 0 < stride <= chunk_size):
            raise ValueError("Require integer 0 < stride <= chunk_size")
        self.chunk_size = chunk_size
        self.stride = stride

    def chunk(self, text: str, metadata: Metadata | None = None) -> list[TextChunk]:
        chunks = []
        for start in range(0, len(text), self.stride):
            end = min(start + self.chunk_size, len(text))
            chunks.append(TextChunk(len(chunks), start, end, text[start:end],
                                    dict(metadata or {})))
            if end == len(text):
                break
        return chunks
```

Each processor delegates chunking. The JSON adapter translates serialized content into one string; the registry factory is the only selection point. The CLI remains generic.

### `src/ragworkshop/processors/formats.py`

```python
import json
from collections.abc import Callable
from typing import Protocol

from ragworkshop.processors.chunks import Metadata, TextChunk, TextChunker


class TextProcessor(Protocol):
    def process(self, text: str, metadata: Metadata | None = None) -> list[TextChunk]: ...


class PlainTextProcessor:
    def __init__(self, chunker: TextChunker):
        self.chunker = chunker

    def process(self, text: str, metadata: Metadata | None = None) -> list[TextChunk]:
        return self.chunker.chunk(text, metadata)


class MarkdownTextProcessor:
    """Preserve Markdown syntax; offsets refer to the Markdown source."""
    def __init__(self, chunker: TextChunker):
        self.chunker = chunker

    def process(self, text: str, metadata: Metadata | None = None) -> list[TextChunk]:
        return self.chunker.chunk(text, {**(metadata or {}), "format": "markdown"})


class JsonTextProcessor:
    def __init__(self, chunker: TextChunker, field: str = "text"):
        self.chunker = chunker
        self.field = field

    def process(self, text: str, metadata: Metadata | None = None) -> list[TextChunk]:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON text") from exc
        if not isinstance(value, dict) or not isinstance(value.get(self.field), str):
            raise ValueError("JSON field must contain a string")
        return self.chunker.chunk(value[self.field],
                                  {**(metadata or {}), "field": self.field})


Builder = Callable[[TextChunker, str], TextProcessor]
PROCESSORS: dict[str, Builder] = {
    "plain": lambda chunker, field: PlainTextProcessor(chunker),
    "markdown": lambda chunker, field: MarkdownTextProcessor(chunker),
    "json": lambda chunker, field: JsonTextProcessor(chunker, field),
}


def get_processor(name: str, chunker: TextChunker, field: str = "text") -> TextProcessor:
    try:
        builder = PROCESSORS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported text format: {name}") from exc
    return builder(chunker, field)
```

The Protocol specifies behavior without requiring inheritance. Implementations and lightweight fakes must preserve this contract. Validation belongs at boundaries so malformed provider output never reaches persistence.

### `src/ragworkshop/embeddings/base.py`

```python
import math
from typing import Protocol


class EmbeddingProvider(Protocol):
    model_name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def validate_vectors(vectors: list[list[float]], count: int, dimensions: int) -> None:
    if len(vectors) != count:
        raise ValueError("Embedding count mismatch")
    if type(dimensions) is not int or dimensions < 1:
        raise ValueError("Invalid embedding dimensions")
    for vector in vectors:
        if len(vector) != dimensions:
            raise ValueError("Embedding dimension mismatch")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in vector):
            raise ValueError("Embedding values must be finite numbers")
        if not any(vector):
            raise ValueError("Zero vectors cannot be used for cosine search")
```

This fixed vocabulary produces stable nonzero vectors across processes. It intentionally makes no semantic-quality claim.

### `src/ragworkshop/embeddings/fake.py`

```python
import re


class FakeEmbeddingProvider:
    """Deterministic lexical demo, NOT a semantic embedding model."""
    model_name = "demo-keyword-v1"
    dimensions = 8
    vocabulary = ("python", "cat", "dog", "database", "storage", "text", "vector")

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            words = re.findall(r"\w+", text.casefold())
            vectors.append([float(words.count(word)) for word in self.vocabulary] + [1.0])
        return vectors
```

The concrete adapter owns external resource handling and validation. Its private helpers centralize invariants so public methods remain small. Inspect the module-specific discussion for the atomic-write or model-response policy.

### `src/ragworkshop/embeddings/local.py`

```python
import hashlib
from pathlib import Path

from ragworkshop.embeddings.base import validate_vectors


class LocalEmbeddingProvider:
    def __init__(self, model_path: str, dimensions: int):
        path = Path(model_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("Model must be a regular file")
        if type(dimensions) is not int or dimensions < 1:
            raise ValueError("Invalid embedding dimensions")
        from llama_cpp import Llama
        with path.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        self.model_name = f"gguf:{fingerprint}:raw-v1"
        self.dimensions = dimensions
        self._model = Llama(model_path=str(path), embedding=True, n_gpu_layers=0,
                            n_ctx=2048, verbose=False)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._model.create_embedding(texts)
        data = sorted(response["data"], key=lambda item: item["index"])
        if [item["index"] for item in data] != list(range(len(texts))):
            raise ValueError("Missing or duplicate embedding index")
        vectors = [item["embedding"] for item in data]
        validate_vectors(vectors, len(texts), self.dimensions)
        return vectors

    def close(self) -> None:
        self._model.close()
```

A simple factory constructs a supported adapter using explicit arguments and rejects unknown names. It does not reach into global settings.

### `src/ragworkshop/embeddings/factory.py`

```python
from ragworkshop.embeddings.base import EmbeddingProvider
from ragworkshop.embeddings.fake import FakeEmbeddingProvider


def get_embedding_provider(name: str = "fake", model_path: str = "",
                           dimensions: int = 8) -> EmbeddingProvider:
    if name == "fake":
        if dimensions != 8:
            raise ValueError("Fake provider requires 8 dimensions")
        return FakeEmbeddingProvider()
    if name == "local":
        if not model_path:
            raise ValueError("Local provider requires an embedding model path")
        from ragworkshop.embeddings.local import LocalEmbeddingProvider
        return LocalEmbeddingProvider(model_path, dimensions)
    raise ValueError(f"Unsupported embedding provider: {name}")
```

The operation batches chunks and preserves order. It receives a provider rather than constructing one and validates every batch.

### `src/ragworkshop/operations/embedding.py`

```python
from ragworkshop.embeddings.base import EmbeddingProvider, validate_vectors
from ragworkshop.processors.chunks import TextChunk


def embed_chunks(chunks: list[TextChunk], provider: EmbeddingProvider,
                 batch_size: int = 16) -> list[list[float]]:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    vectors = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        output = provider.embed([chunk.text for chunk in batch])
        validate_vectors(output, len(batch), provider.dimensions)
        vectors.extend(output)
    return vectors
```

## Refactor, pattern rationale, and alternatives

An ABC can enforce abstract-method implementation at instantiation; a Protocol makes tiny fakes easier and avoids forced inheritance. Sentence or token chunkers can replace the character chunker after defining offset semantics. A richer Markdown parser needs declared dependencies and explicit handling of embedded HTML; regular-expression stripping is not a complete parser.

Chunking belongs to `TextChunker`, not format adapters, loaders, or embedding providers. We use characters as Python Unicode code points, not bytes, grapheme clusters, or model tokens. With size 4 and stride 3, `ABCDEFGHIJ` becomes `ABCD`, `DEFG`, `GHIJ`. Overlap is `4 - 3 = 1`. Size 4 and stride 4 do not overlap. A stride larger than size would lose content, so this contract rejects it.

No implicit case-folding, whitespace collapse, or Unicode normalization changes source offsets. An embedding strategy can prepare its own inputs, but chunk text and provenance remain unchanged. Normalization can improve consistency but needs an explicit offset mapping if the UI must highlight original text. Character windows can split words and exceed a model token budget; token-aware or sentence-aware chunking is a stretch exercise. Large chunks retain context but may dilute a match; very small chunks lose context; overlap improves continuity at additional storage and inference cost.

`TextProcessor` is a structural Protocol: an object conforms by providing `process`, without inheritance. Plain, Markdown, and JSON adapters participate as strategies. Each delegates to the same chunker. Markdown remains Markdown source, including headings and punctuation; this is intentionally not a full Markdown-to-plain-text renderer. JSON selects one top-level string field and preserves caller-supplied source metadata. Its offsets refer to the extracted field, not the serialized JSON. Physical DOCX/PDF parsing is a separate concern introduced next module.

`PROCESSORS` is a small registry; `get_processor` is a simple factory, not the GoF subclass-based Factory Method pattern. Adding a registry entry changes construction configuration, while the pipeline and CLI stay closed to new format-specific branches. This is a pragmatic application of OCP, not a promise that no file ever changes.

`EmbeddingProvider.embed(list[str])` returns one finite nonzero vector per input, in input order, with declared dimensionality. Its identity describes an embedding space; equal vector length alone does not mean two models are compatible. `embed_chunks` batches calls and validates each batch. Provider failures propagate rather than silently dropping chunks.

`FakeEmbeddingProvider` counts seven fixed words plus a constant coordinate. It is stable across Python processes, unlike Python's randomized `hash()`. It demonstrates routing and retrieval mechanics, not semantic understanding. `LocalEmbeddingProvider` adapts `llama-cpp-python`, lazily importing it and sorting indexed responses. Its model fingerprint and preprocessing version identify the space. Computing the fingerprint costs a file read but avoids trusting a mutable filename.

The local example uses raw, symmetric input text. Choose an embedding-capable GGUF supporting that usage and configure its actual dimension; arbitrary generation models are unsuitable. Model-specific document/query prefixes and pooling must be implemented as a coherent provider policy and given a different identity. Do not paste one model's prefixes into general operations. Real inference is CPU-only here and loaded once per provider; the optional dependency avoids making a native build a prerequisite for ordinary tests.

## Verification and expected test outcomes

**Working directory: your `rag-workshop/` project root; activate `.venv` first.**

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m pytest -q
rag-workshop info
```

Validated cumulative result: **32 passed, 1 skipped**.

Expected: all implemented tests pass; the real-GGUF test is skipped unless explicitly opted in. This skip is expected, not evidence that real inference was tested.

**Working directory: `rag-workshop/`.**

```bash
rag-workshop process ABCDEFGHIJ --chunk-size 4 --stride 3
# chunks: ABCD at [0,4), DEFG at [3,7), GHIJ at [6,10)
rag-workshop process '{"body":"python"}' --format json --field body
# one chunk with text "python" and metadata {"field":"body"}
rag-workshop embed python
# model: demo-keyword-v1; vectors: [[1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]]
```

## Edge cases and limitations

Empty input yields no chunks and no provider calls. Repeated content is not deduplicated. Boolean and fractional sizes are rejected. Missing/duplicate model response indexes, inconsistent dimensions, zero vectors, NaN, and provider exceptions must not produce silently truncated output. Markdown offsets and JSON-field offsets have intentionally different source representations.

## Solution architecture trace

Arrows show **data transformations** for a concrete JSON example. This is a trace of the strategy/factory collaboration, not a second chunking implementation.

```text
get_processor("json", chunker, field="body")
                |
                v
JsonTextProcessor receives:
  text = '{"body": "python"}'
  metadata = {"source": "memo"}
                |
                v
selected string = "python"
metadata = {"source": "memo", "field": "body"}
                |
       shared chunker, size 30, stride 30
                v
one TextChunk: index 0, start 0, end 6, text "python"
                |
       embed_chunks with FakeEmbeddingProvider
                v
[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
```

The offsets belong to `python`, not to the surrounding JSON. The first vector coordinate counts the fixed vocabulary word `python`; the last coordinate prevents a zero vector. Factory selection and representation extraction change the input to the shared chunker without changing batching or provider validation. This output proves the fake pipeline contract, not semantic quality.

## Why the acceptance criteria are met

- Implement PlainTextProcessor.process using the shared chunker. The completed code and assertions above make this behavior observable.
- Implement MarkdownTextProcessor.process while preserving Markdown syntax and source metadata. The completed code and assertions above make this behavior observable.
- Implement JsonTextProcessor.process with configurable top-level field; reject malformed JSON, non-object JSON, missing fields, and non-string fields. The completed code and assertions above make this behavior observable.
- Implement get_processor using the provided registry; demonstrate a custom registry entry without changing the CLI. The completed code and assertions above make this behavior observable.
- Run fake-provider pipeline tests and verify chunk offsets refer to the selected representation. The completed code and assertions above make this behavior observable.

Tests are the executable acceptance evidence; successful imports alone are insufficient. Exact validation counts for all checkpoints are recorded in the workshop README. The optional real-model check is reported separately from default offline validation.
