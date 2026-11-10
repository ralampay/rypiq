# Activity 2: Text handling and embedding abstraction

## Learning outcome

Validate vectors and distinguish offline adapter tests from opt-in model tests.

## Problem and motivation

Incoming content can be plain text, Markdown source, or JSON with a configurable text field. Add interchangeable adapters so the CLI and embedding pipeline do not accumulate format-specific conditionals.

## Prerequisites and starter setup

Complete Module 1 and keep its source and tests. Start from that checkpoint, never from the final application.

**Working directory: `~/rag-workshop/`, with `.venv` active.** `WORKSHOP_MATERIALS` is the absolute materials directory exported in Module 1. This overlay changes only this module's files; it preserves earlier completed exercises. It supplies infrastructure and replaces the exercise bodies with TODOs. Commit or copy your previous checkpoint first if you want to compare it later.

```bash
cp -R "$WORKSHOP_MATERIALS/starters/module-02/." .
python -m pip install -e ".[dev]"
```

Read the supplied scaffolding and write the tests below before filling any TODO. The overlay includes those completed test files as an instructor recovery aid; learners can retype or extend them before implementing behavior. Tests from previous modules remain regression checks.

**Files introduced or replaced in this module:** unchanged files from previous modules remain in place. A test may import a newly introduced TODO module and fail; that is intentional Red, not a dependency on a later module.

```text
rag-workshop/
  src/ragworkshop/cli.py
  src/ragworkshop/processors/chunks.py
  src/ragworkshop/processors/formats.py
  src/ragworkshop/embeddings/base.py
  src/ragworkshop/embeddings/fake.py
  src/ragworkshop/embeddings/local.py
  src/ragworkshop/embeddings/factory.py
  src/ragworkshop/operations/embedding.py
  tests/unit/test_text.py
  tests/unit/test_local.py
  tests/integration/test_real_model.py
  tests/integration/test_text_cli.py
```

## Required tasks

1. Implement PlainTextProcessor.process using the shared chunker.
2. Implement MarkdownTextProcessor.process while preserving Markdown syntax and source metadata.
3. Implement JsonTextProcessor.process with configurable top-level field; reject malformed JSON, non-object JSON, missing fields, and non-string fields.
4. Implement get_processor using the provided registry; demonstrate a custom registry entry without changing the CLI.
5. Run fake-provider pipeline tests and verify chunk offsets refer to the selected representation.

## Exact exercise files and interface stubs

These are the files to create or replace. Bodies marked TODO are deliberately incomplete. All other changed files in the tree are supplied scaffolding, not hidden requirements. The stage overlay contains exactly these stubs and the tests below. Keep public signatures unchanged.

### `src/ragworkshop/processors/formats.py`

```python
import json
from collections.abc import Callable
from typing import Protocol
from ragworkshop.processors.chunks import Metadata, TextChunk, TextChunker

class TextProcessor(Protocol):

    def process(self, text: str, metadata: Metadata | None=None) -> list[TextChunk]:
        ...

class PlainTextProcessor:

    def __init__(self, chunker: TextChunker):
        self.chunker = chunker

    def process(self, text: str, metadata: Metadata | None=None) -> list[TextChunk]:
        raise NotImplementedError('TODO: PlainTextProcessor.process')

class MarkdownTextProcessor:
    """Preserve Markdown syntax; offsets refer to the Markdown source."""

    def __init__(self, chunker: TextChunker):
        self.chunker = chunker

    def process(self, text: str, metadata: Metadata | None=None) -> list[TextChunk]:
        raise NotImplementedError('TODO: MarkdownTextProcessor.process')

class JsonTextProcessor:

    def __init__(self, chunker: TextChunker, field: str='text'):
        self.chunker = chunker
        self.field = field

    def process(self, text: str, metadata: Metadata | None=None) -> list[TextChunk]:
        raise NotImplementedError('TODO: JsonTextProcessor.process')
Builder = Callable[[TextChunker, str], TextProcessor]
PROCESSORS: dict[str, Builder] = {'plain': lambda chunker, field: PlainTextProcessor(chunker), 'markdown': lambda chunker, field: MarkdownTextProcessor(chunker), 'json': lambda chunker, field: JsonTextProcessor(chunker, field)}

def get_processor(name: str, chunker: TextChunker, field: str='text') -> TextProcessor:
    raise NotImplementedError('TODO: get_processor')
```

## Red — tests to write or complete

Write these tests before implementing the TODOs. You may use the supplied test files as a starting point; add one edge-case assertion of your own. Do not weaken assertions or mark unfinished behavior skipped.

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

## Green and Refactor constraints

- Implement PlainTextProcessor.process using the shared chunker.
- Implement MarkdownTextProcessor.process while preserving Markdown syntax and source metadata.
- Implement JsonTextProcessor.process with configurable top-level field; reject malformed JSON, non-object JSON, missing fields, and non-string fields.
- Implement get_processor using the provided registry; demonstrate a custom registry entry without changing the CLI.
- Run fake-provider pipeline tests and verify chunk offsets refer to the selected representation.
- First satisfy one failing assertion, then rerun the current module tests.
- Refactor only after Green; preserve earlier commands and tests.
- Keep business behavior out of CLI branches. Use supplied interfaces and explicit arguments.
- Keep automated tests offline, with temporary paths for file/database interactions.
- No GPU, cloud credentials, or real GGUF is required.

## Input/output examples and expected observable behavior

**Working directory: `rag-workshop/`.** Output comments specify assertions, not byte-for-byte JSON formatting.

```bash
rag-workshop process ABCDEFGHIJ --chunk-size 4 --stride 3
# chunks: ABCD at [0,4), DEFG at [3,7), GHIJ at [6,10)
rag-workshop process '{"body":"python"}' --format json --field body
# one chunk with text "python" and metadata {"field":"body"}
rag-workshop embed python
# model: demo-keyword-v1; vectors: [[1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0]]
```

## Acceptance criteria

- Implement PlainTextProcessor.process using the shared chunker.
- Implement MarkdownTextProcessor.process while preserving Markdown syntax and source metadata.
- Implement JsonTextProcessor.process with configurable top-level field; reject malformed JSON, non-object JSON, missing fields, and non-string fields.
- Implement get_processor using the provided registry; demonstrate a custom registry entry without changing the CLI.
- Run fake-provider pipeline tests and verify chunk offsets refer to the selected representation.
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

Validated cumulative result: **32 passed, 1 skipped**.

Expected: all implemented tests pass; the real-GGUF test is skipped unless explicitly opted in. This skip is expected, not evidence that real inference was tested.

## Optional stretch challenges

- Register a fourth textual representation without editing CLI logic. Add tests that prove metadata and ordering remain intact.
- Explore token-aware chunking, but define how offsets map back to text before coding.
