# Introduction

Build a small Python application that converts a protocol document into embeddings using **EmbeddingGemma 300M**. A protocol is a UTF-8 text file describing a procedure. Each chunk will retain its text and source offsets alongside its embedding.

Follow a test-first workflow: write a test, run it and observe the failure, implement the behavior, then run the test again.

# Objectives

- Build and unit-test `LocalEmbeddingProvider` without loading a real model.
- Build processors for reading text and creating overlapping chunks.
- Build operations that prepare model inputs and attach embeddings to chunks.
- Add a CLI and test the complete path from arguments to JSON output.

# Workflow

Use Python 3.10 or newer. Run commands from your tutorial project directory. Create this package structure:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install pytest
mkdir -p protocol_embeddings tests/unit tests/integration models
touch protocol_embeddings/__init__.py
```

Unit tests and the first CLI integration tests replace the inference runtime, so they need no model or `llama-cpp-python` installation. Install the runtime and provide a GGUF model for the final real-model check.

## Section 1

Build the local embedding provider first. It loads a model once and returns one finite, nonempty vector per input, in input order.

### Tests to Write

Cover model loading, missing files, empty input, model reuse, response ordering, vector counts, duplicate indices, inconsistent dimensions, and invalid values.

Save as `tests/unit/test_local_provider.py`:

```python
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from protocol_embeddings.local import LocalEmbeddingProvider


@pytest.fixture
def runtime(monkeypatch, tmp_path):
    model = tmp_path / "model.gguf"
    model.touch()
    engine = Mock()
    engine.create_embedding.return_value = {
        "data": [{"index": 0, "embedding": [0.1, 0.2]}]
    }
    constructor = Mock(return_value=engine)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=constructor))
    return model, constructor, engine


def test_load_and_reuse_model(runtime):
    model, constructor, engine = runtime
    provider = LocalEmbeddingProvider(str(model), context_size=2048, gpu_layers=0)
    assert provider.model_name == "model.gguf"
    assert provider.embed(["first"]) == [[0.1, 0.2]]
    assert provider.embed(["second"]) == [[0.1, 0.2]]
    constructor.assert_called_once_with(
        model_path=str(model.resolve()), embedding=True,
        n_ctx=2048, n_gpu_layers=0, verbose=False,
    )
    assert engine.create_embedding.call_count == 2
    engine.create_embedding.assert_called_with(["second"])


def test_missing_model(runtime, tmp_path):
    _, constructor, _ = runtime
    with pytest.raises(FileNotFoundError):
        LocalEmbeddingProvider(str(tmp_path / "missing.gguf"))
    constructor.assert_not_called()


def test_empty_input(runtime):
    model, _, engine = runtime
    assert LocalEmbeddingProvider(str(model)).embed([]) == []
    engine.create_embedding.assert_not_called()


def test_response_is_sorted_by_input_index(runtime):
    model, _, engine = runtime
    engine.create_embedding.return_value = {"data": [
        {"index": 1, "embedding": [0.3, 0.4]},
        {"index": 0, "embedding": [0.1, 0.2]},
    ]}
    assert LocalEmbeddingProvider(str(model)).embed(["a", "b"]) == [
        [0.1, 0.2], [0.3, 0.4],
    ]
    engine.create_embedding.assert_called_once_with(["a", "b"])


@pytest.mark.parametrize("data", [
    [],  # Wrong vector count.
    [{"index": 0, "embedding": [0.1]}] * 2,  # Duplicate index.
    [{"index": 0, "embedding": [0.1]},
     {"index": 2, "embedding": [0.2]}],  # Missing index 1.
    [{"index": 0, "embedding": [0.1]},
     {"index": 1, "embedding": [0.2, 0.3]}],  # Different dimensions.
])
def test_invalid_response(runtime, data):
    model, _, engine = runtime
    engine.create_embedding.return_value = {"data": data}
    with pytest.raises(ValueError):
        LocalEmbeddingProvider(str(model)).embed(["a", "b"])


@pytest.mark.parametrize("vector", [
    [], [float("nan")], [float("inf")], [True], ["bad"], [[0.1]],
])
def test_invalid_vector(runtime, vector):
    model, _, engine = runtime
    engine.create_embedding.return_value = {
        "data": [{"index": 0, "embedding": vector}]
    }
    with pytest.raises(ValueError):
        LocalEmbeddingProvider(str(model)).embed(["a"])
```

Run `python -m pytest -q tests/unit/test_local_provider.py`. The first run should fail because the provider module has not been created.

### Implementation

Import the inference runtime inside the constructor. This lets tests substitute a fake runtime before construction. Keep model-specific text formatting outside this provider.

Save as `protocol_embeddings/local.py`:

```python
import math
from pathlib import Path


class LocalEmbeddingProvider:
    def __init__(self, model_path, context_size=2048, gpu_layers=0):
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Embedding model not found: {path}")

        from llama_cpp import Llama

        self.model_name = path.name
        self._model = Llama(
            model_path=str(path), embedding=True,
            n_ctx=context_size, n_gpu_layers=gpu_layers, verbose=False,
        )

    def embed(self, texts):
        if not texts:
            return []
        response = self._model.create_embedding(texts)
        rows = sorted(response["data"], key=lambda row: row["index"])
        if len(rows) != len(texts):
            raise ValueError("Embedding count mismatch")

        vectors = []
        for index, row in enumerate(rows):
            if row["index"] != index:
                raise ValueError("Missing or duplicate embedding index")
            vector = row["embedding"]
            if not isinstance(vector, list) or not vector:
                raise ValueError("Expected a nonempty vector")
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in vector
            ):
                raise ValueError("Invalid embedding value")
            if vectors and len(vector) != len(vectors[0]):
                raise ValueError("Inconsistent vector dimensions")
            vectors.append(vector)
        return vectors
```

Run the provider tests again. They should now pass.

## Section 2

Build the text processors, then the operations that connect chunks to the model.

### Tests to Write

Start with file reading and chunk boundaries. Offsets use Python string slicing: `start` is inclusive and `end` is exclusive.

Save as `tests/unit/test_processors.py`:

```python
import pytest

from protocol_embeddings.processors import read_text_file, chunk_text


def test_read_utf8(tmp_path):
    path = tmp_path / "protocol.txt"
    path.write_text("Measure 20 °C.", encoding="utf-8")
    assert read_text_file(path) == "Measure 20 °C."


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_text_file(tmp_path / "missing.txt")


def test_unsupported_extension(tmp_path):
    with pytest.raises(ValueError, match=".txt"):
        read_text_file(tmp_path / "protocol.pdf")


def test_chunks_preserve_offsets():
    assert chunk_text("ABCDEFGHIJ", 4, 3) == [
        {"index": 0, "start": 0, "end": 4, "text": "ABCD"},
        {"index": 1, "start": 3, "end": 7, "text": "DEFG"},
        {"index": 2, "start": 6, "end": 10, "text": "GHIJ"},
    ]
    assert chunk_text("", 4, 3) == []
    assert chunk_text("AB", 4, 3) == [
        {"index": 0, "start": 0, "end": 2, "text": "AB"},
    ]


@pytest.mark.parametrize("size,stride", [(0, 1), (4, 0), (4, 5), (True, 1), (4, 1.5)])
def test_invalid_chunk_parameters(size, stride):
    with pytest.raises(ValueError):
        chunk_text("text", size, stride)
```

Next, test the operations with a fake provider. Verify the document prefix, batching, unchanged source text, and skipping blank chunks before loading a model.

Save as `tests/unit/test_operations.py`:

```python
from unittest.mock import Mock

import pytest

from protocol_embeddings.operations import process_text_file, embed_text_file


def test_process_file(tmp_path):
    path = tmp_path / "protocol.txt"
    path.write_text("ABCDE", encoding="utf-8")
    result = process_text_file(path, 4, 3)
    assert result["source"] == str(path)
    assert result["chunk_count"] == 2
    assert result["chunks"][-1]["text"] == "DE"


def test_embed_batches(tmp_path):
    path = tmp_path / "protocol.txt"
    path.write_text("ABCDEFGHIJ", encoding="utf-8")
    provider = Mock(model_name="fake.gguf")
    provider.embed.side_effect = lambda texts: [[0.1, 0.2] for _ in texts]
    factory = Mock(return_value=provider)

    result = embed_text_file(path, 4, 3, "fake.gguf", batch_size=2,
                             provider_factory=factory)

    factory.assert_called_once_with(
        model_path="fake.gguf", context_size=2048, gpu_layers=0,
    )
    assert [call.args[0] for call in provider.embed.call_args_list] == [
        ["title: none | text: ABCD", "title: none | text: DEFG"],
        ["title: none | text: GHIJ"],
    ]
    assert result["dimensions"] == 2
    assert result["model"] == "fake.gguf"
    assert result["embedded_chunk_count"] == 3
    assert result["chunks"][0] == {
        "index": 0, "start": 0, "end": 4, "text": "ABCD",
        "embedding": [0.1, 0.2],
    }


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_protocol_does_not_load_model(tmp_path, text):
    path = tmp_path / "protocol.txt"
    path.write_text(text, encoding="utf-8")
    factory = Mock()
    with pytest.raises(ValueError, match="nonempty"):
        embed_text_file(path, 4, 3, "fake.gguf", provider_factory=factory)
    factory.assert_not_called()


def test_skip_blank_chunk(tmp_path):
    path = tmp_path / "protocol.txt"
    path.write_text("    ABCD", encoding="utf-8")
    provider = Mock(model_name="fake.gguf")
    provider.embed.return_value = [[0.1]]
    result = embed_text_file(path, 4, 4, "fake.gguf",
                             provider_factory=Mock(return_value=provider))
    assert result["chunk_count"] == 2
    assert result["embedded_chunk_count"] == 1
    assert result["chunks"][0]["index"] == 1


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_invalid_batch_size(batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        embed_text_file("unused.txt", 4, 3, "unused.gguf", batch_size=batch_size)


@pytest.mark.parametrize("batches", [
    [[], [[0.2]]],  # Wrong count.
    [[[0.1]], [[0.2, 0.3]]],  # Dimensions change between batches.
])
def test_invalid_batch_results(tmp_path, batches):
    path = tmp_path / "protocol.txt"
    path.write_text("ABCDE", encoding="utf-8")
    provider = Mock(model_name="fake.gguf")
    provider.embed.side_effect = batches
    with pytest.raises(ValueError):
        embed_text_file(path, 4, 3, "fake.gguf",
                        provider_factory=Mock(return_value=provider))
```

Run `python -m pytest -q tests/unit`. Observe the missing modules before implementing them.

### Implementation

Keep reading and chunking separate so each processor has a small, testable responsibility.

Save as `protocol_embeddings/processors.py`:

```python
from pathlib import Path


def read_text_file(path):
    path = Path(path)
    if path.suffix.lower() != ".txt":
        raise ValueError("Only .txt files are supported")
    return path.read_text(encoding="utf-8")


def chunk_text(text, chunk_size, stride):
    if (type(chunk_size) is not int or type(stride) is not int
            or not 0 < stride <= chunk_size):
        raise ValueError("Require positive integers with stride <= chunk_size")
    chunks = []
    for start in range(0, len(text), stride):
        end = min(start + chunk_size, len(text))
        chunks.append({
            "index": len(chunks), "start": start, "end": end,
            "text": text[start:end],
        })
        if end == len(text):
            break
    return chunks
```

Compose the processors in `process_text_file()`, then add inference in `embed_text_file()`. Injecting the provider factory keeps operation tests independent of the runtime.

Save as `protocol_embeddings/operations.py`:

```python
from protocol_embeddings.local import LocalEmbeddingProvider
from protocol_embeddings.processors import read_text_file, chunk_text

DOCUMENT_PREFIX = "title: none | text: "


def process_text_file(path, chunk_size, stride):
    text = read_text_file(path)
    chunks = chunk_text(text, chunk_size, stride)
    return {
        "source": str(path), "chunk_count": len(chunks), "chunks": chunks,
    }


def embed_text_file(path, chunk_size, stride, model_path,
                    context_size=2048, gpu_layers=0, batch_size=1,
                    provider_factory=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    document = process_text_file(path, chunk_size, stride)
    chunks = [chunk for chunk in document["chunks"] if chunk["text"].strip()]
    if not chunks:
        raise ValueError("Protocol has no nonempty chunks")

    factory = provider_factory if provider_factory is not None else LocalEmbeddingProvider
    provider = factory(model_path=model_path, context_size=context_size,
                       gpu_layers=gpu_layers)
    records = []
    dimensions = None
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors = provider.embed([DOCUMENT_PREFIX + chunk["text"] for chunk in batch])
        if len(vectors) != len(batch):
            raise ValueError("Embedding count mismatch")
        for chunk, vector in zip(batch, vectors):
            if not vector:
                raise ValueError("Empty embedding vector")
            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise ValueError("Inconsistent vector dimensions")
            records.append({**chunk, "embedding": vector})
    return {
        **document, "model": provider.model_name, "dimensions": dimensions,
        "embedded_chunk_count": len(records), "chunks": records,
    }
```

Run `python -m pytest -q tests/unit` again.

Chunk sizes and strides count **characters**, while model context counts **tokens**. A size of 500 and stride of 400 gives full neighboring chunks 100 characters of overlap. Keep each prefixed input within the model's token limit; this simple chunker does not enforce token budgets.

## Section 3

Add the CLI and test the components together.

### Tests to Write

The following integration tests exercise argument parsing, file reading, chunking, the real provider adapter, and JSON serialization. Only the inference engine is replaced. This checks component wiring without downloading weights.

Save as `tests/integration/test_cli.py`:

```python
import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from protocol_embeddings.cli import main


@pytest.mark.parametrize("write_file", [False, True])
def test_cli_generates_embeddings(tmp_path, monkeypatch, capsys, write_file):
    protocol = tmp_path / "protocol.txt"
    protocol.write_text("ABCDEFGHIJ", encoding="utf-8")
    model = tmp_path / "model.gguf"
    model.touch()
    engine = Mock()
    engine.create_embedding.side_effect = lambda texts: {"data": [
        {"index": i, "embedding": [0.1, 0.2]} for i, _ in enumerate(texts)
    ]}
    constructor = Mock(return_value=engine)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=constructor))
    args = [str(protocol), "--model", str(model),
            "--chunk-size", "4", "--stride", "3", "--batch-size", "2"]
    output = tmp_path / "output.json"
    if write_file:
        args += ["--output", str(output)]

    main(args)

    stdout = capsys.readouterr().out
    result = json.loads(output.read_text() if write_file else stdout)
    assert result["source"] == str(protocol)
    assert result["dimensions"] == 2
    assert result["embedded_chunk_count"] == 3
    assert [chunk["text"] for chunk in result["chunks"]] == ["ABCD", "DEFG", "GHIJ"]
    assert result["chunks"][1]["start"] == 3
    assert result["chunks"][0]["embedding"] == [0.1, 0.2]
    constructor.assert_called_once()
    assert engine.create_embedding.call_args_list[0].args[0] == [
        "title: none | text: ABCD", "title: none | text: DEFG",
    ]
    if write_file:
        assert stdout == ""


def test_cli_reports_missing_protocol(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main([str(tmp_path / "missing.txt"), "--model", "unused.gguf"])
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
```

Run `python -m pytest -q tests/integration/test_cli.py` before adding the CLI module.

### Implementation

Accept an optional argument list in `main()` so tests can call the same entry point used by the terminal.

Save as `protocol_embeddings/cli.py`:

```python
import argparse
import json
from pathlib import Path

from protocol_embeddings.operations import embed_text_file


def main(argv=None):
    parser = argparse.ArgumentParser(description="Embed a protocol text file")
    parser.add_argument("path")
    parser.add_argument("--model", required=True)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--stride", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        result = embed_text_file(
            args.path, args.chunk_size, args.stride, args.model,
            batch_size=args.batch_size,
        )
        serialized = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
        if args.output:
            Path(args.output).write_text(serialized + "\n", encoding="utf-8")
        else:
            print(serialized)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
```

Run all tests:

```bash
python -m pytest -q tests
```

For a real-model check, install the inference runtime and place an EmbeddingGemma 300M GGUF file at `models/embeddinggemma-300m-Q4_0.gguf`, or adjust the path below. The model produces 768-dimensional embeddings with a maximum input context of 2,048 tokens. See the [official model card](https://huggingface.co/google/embeddinggemma-300m) for its input formatting and model details.

```bash
python -m pip install llama-cpp-python
cat > protocol.txt <<'TEXT'
Record the sample identifier.
Check the instrument calibration before processing.
Save the measurement and its timestamp.
TEXT

python -m protocol_embeddings.cli protocol.txt \
  --model models/embeddinggemma-300m-Q4_0.gguf \
  --chunk-size 500 \
  --stride 400 \
  --output protocol-embeddings.json

python - <<'PYCODE'
import json
import math
from pathlib import Path

result = json.loads(Path("protocol-embeddings.json").read_text(encoding="utf-8"))
assert result["dimensions"] == 768
assert result["embedded_chunk_count"] == len(result["chunks"]) > 0
for chunk in result["chunks"]:
    assert len(chunk["embedding"]) == 768
    assert all(math.isfinite(value) for value in chunk["embedding"])
print("Verified", result["embedded_chunk_count"], "embeddings")
PYCODE
```

The output file contains the source path, model name, dimensions, and chunk records. Each record contains original text, offsets, and an embedding. `chunk_count` includes blank chunks; `embedded_chunk_count` excludes them. Reusing an output filename replaces its contents. This walkthrough ends with JSON output; it does not require a vector database.

