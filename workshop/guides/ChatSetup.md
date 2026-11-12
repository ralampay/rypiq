# Introduction

Send one query to a local language model and print its response:

```bash
rypiq chat --model /path/to/model.gguf --query "Some text query"
```

Each command loads the model, answers the query, and exits. There is no conversation history or memory. Standard output contains only the response.

Use a chat/instruction-tuned GGUF model supported by `llama-cpp-python`. EmbeddingGemma produces vectors and cannot act as the chat model. This command does not retrieve documents from a database.

# Objectives

- Define an inference provider and use its local implementation to answer one query.
- Test the command without downloading model weights.
- Install the `rypiq` command and try a real query.

# Workflow

Use Python 3.10 or newer. Run commands from your tutorial project directory. Continue with the `protocol_embeddings` package used in the preceding guides, or create it for this exercise:

```bash
python3 -m venv .venv  # Skip if your tutorial environment already exists.
source .venv/bin/activate
python -m pip install pytest
mkdir -p protocol_embeddings/inference tests/integration
touch protocol_embeddings/__init__.py protocol_embeddings/inference/__init__.py
```

The automated tests need no model weights or inference runtime. Each labelled Python block is the complete contents of that file.

## Section 1: Test the command

Save as `tests/integration/test_chat_cli.py`:

```python
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from protocol_embeddings.chat_cli import main


@pytest.fixture
def chat_model(tmp_path, monkeypatch):
    path = tmp_path / "chat.gguf"
    path.touch()
    engine = Mock()
    engine.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Hello!"}}]
    }
    constructor = Mock(return_value=engine)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=constructor))
    return path, engine, constructor


def test_single_query(chat_model, capsys):
    path, engine, constructor = chat_model
    main(["chat", "--model", str(path), "--query", "  Hello  "])

    assert capsys.readouterr().out == "Hello!\n"
    constructor.assert_called_once_with(
        model_path=str(path.resolve()), n_ctx=4096,
        n_gpu_layers=0, verbose=False,
    )
    engine.create_chat_completion.assert_called_once_with(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=512, temperature=0.7, stream=False,
    )
    engine.close.assert_called_once()


def test_separate_queries_have_no_history(chat_model):
    path, engine, constructor = chat_model
    main(["chat", "--model", str(path), "--query", "My name is Sam."])
    main(["chat", "--model", str(path), "--query", "What is my name?"])

    assert constructor.call_count == 2
    assert engine.create_chat_completion.call_args.kwargs["messages"] == [
        {"role": "user", "content": "What is my name?"}
    ]
    assert engine.close.call_count == 2


def test_provider_reuse_has_no_history(chat_model):
    from protocol_embeddings.inference.base import InferenceProvider
    from protocol_embeddings.inference.local import LocalInferenceProvider

    path, engine, constructor = chat_model
    provider = LocalInferenceProvider(str(path))
    try:
        assert isinstance(provider, InferenceProvider)
        assert provider.model_name == "chat.gguf"
        assert provider.infer("My name is Sam.") == "Hello!"
        assert provider.infer("What is my name?") == "Hello!"
        assert engine.create_chat_completion.call_args.kwargs["messages"] == [
            {"role": "user", "content": "What is my name?"}
        ]
        constructor.assert_called_once()
    finally:
        provider.close()
    engine.close.assert_called_once()


@pytest.mark.parametrize("query_args", [[], ["--query", "  "]])
def test_query_required(chat_model, capsys, query_args):
    path, engine, constructor = chat_model
    with pytest.raises(SystemExit) as error:
        main(["chat", "--model", str(path), *query_args])
    assert error.value.code == 2
    assert "query" in capsys.readouterr().err
    constructor.assert_not_called()


def test_missing_model(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(["chat", "--model", str(tmp_path / "missing.gguf"), "--query", "Hello"])
    assert error.value.code == 2
    assert "Chat model not found" in capsys.readouterr().err


def test_generation_failure_closes_model(chat_model, capsys):
    path, engine, _ = chat_model
    engine.create_chat_completion.side_effect = RuntimeError("Generation failed")
    with pytest.raises(SystemExit) as error:
        main(["chat", "--model", str(path), "--query", "Hello"])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "Generation failed" in output.err
    engine.close.assert_called_once()
```

Run `python -m pytest -q tests/integration/test_chat_cli.py`. It should fail because the command module does not exist yet.

## Section 2: Implement the provider and command

Follow the embedding provider pattern: an abstract base class defines the interface, and a local provider wraps the GGUF model. Keep the interface small: a model name, an inference method, and cleanup.

Save as `protocol_embeddings/inference/base.py`:

```python
from abc import ABC, abstractmethod


class InferenceProvider(ABC):
    @property
    @abstractmethod
    def model_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def infer(self, query: str) -> str:
        """Answer one query without conversation history."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
```

Save as `protocol_embeddings/inference/local.py`:

```python
from pathlib import Path

from .base import InferenceProvider


class LocalInferenceProvider(InferenceProvider):
    def __init__(self, model_path: str):
        path = Path(model_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Chat model not found: {path}")

        from llama_cpp import Llama

        self._model_name = path.name
        self._model = Llama(
            model_path=str(path), n_ctx=4096, n_gpu_layers=0, verbose=False,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def infer(self, query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("query cannot be blank")
        response = self._model.create_chat_completion(
            messages=[{"role": "user", "content": query}],
            max_tokens=512, temperature=0.7, stream=False,
        )
        answer = response["choices"][0]["message"]["content"]
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Model returned no text reply")
        return answer.strip()

    def close(self) -> None:
        self._model.close()
```

Each `infer()` call sends only its query, even if the same provider is reused. No messages are stored on the provider.

Save as `protocol_embeddings/chat_cli.py`:

```python
import argparse
import sys

from protocol_embeddings.inference.base import InferenceProvider
from protocol_embeddings.inference.local import LocalInferenceProvider


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Preserve document commands from the preceding guides.
    if argv and argv[0] not in {"chat", "-h", "--help"}:
        from protocol_embeddings.commands import main as document_commands
        return document_commands(argv)

    parser = argparse.ArgumentParser(prog="rypiq", description="Query a local chat model")
    sub = parser.add_subparsers(dest="command", required=True)
    chat = sub.add_parser("chat", help="Answer one query without memory")
    chat.add_argument("--model", required=True, help="Path to a chat GGUF model")
    chat.add_argument("--query", required=True, help="Text to send to the model")
    args = parser.parse_args(argv)
    query = args.query.strip()
    if not query:
        parser.error("query cannot be blank")

    try:
        provider: InferenceProvider = LocalInferenceProvider(args.model)
        try:
            answer = provider.infer(query)
        finally:
            provider.close()
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.error(str(error))
    print(answer)


if __name__ == "__main__":
    main()
```

Run the tests again:

```bash
python -m pytest -q tests/integration/test_chat_cli.py
```

The model receives exactly one user message. Its GGUF chat template determines how that message is formatted. CPU inference is selected with `n_gpu_layers=0`, and the response is printed once generation finishes. Errors go to standard error with a nonzero exit status.

Other commands are forwarded to `protocol_embeddings.commands` from the preceding guides when that module is present. They continue to accept global options before the subcommand.

## Section 3: Install and run

For a standalone tutorial project, save this as `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "protocol-chat-tutorial"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["llama-cpp-python"]

[project.optional-dependencies]
dev = ["pytest"]

[project.scripts]
rypiq = "protocol_embeddings.chat_cli:main"

[tool.setuptools.packages.find]
include = ["protocol_embeddings*"]
```

When continuing a packaged tutorial, merge the script entry and package-discovery setting into its configuration and add the dependencies without replacing its other settings. Keep Chroma installed for document commands. Use the tutorial virtual environment so its `rypiq` executable belongs to this package.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/integration/test_chat_cli.py
rypiq chat --help
rypiq chat --model /path/to/model.gguf --query "What is the capital of France?"
```

Example output (actual wording depends on the model):

```text
The capital of France is Paris.
```

Every invocation starts fresh. To ask another question, run the command again with a new `--query`.
