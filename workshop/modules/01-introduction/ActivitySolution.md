# Activity 1 solution: Introduction and project setup

**Use this file during review or recovery.**

## Match to the exercise

Your team needs a tool that behaves the same outside an IDE and outside its source directory. Package a minimal CLI so installation and import boundaries are verified before adding RAG behavior.

This solution uses exactly the activity signatures and requirements. It is a cumulative change: retain the previous module files and replace/add only the files listed here. The direct link below opens the complete cumulative checkpoint for this module.

[Complete Module 1 checkpoint](../../reference/module-01/)

## Red — completed tests first

Run these against the activity stubs and confirm the TODO failures before implementing. These are the same test contracts as Activity.md, with no changed expectations.

### `tests/conftest.py`

```python
import pytest


def pytest_addoption(parser):
    parser.addoption("--run-real-model", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-real-model"):
        for item in items:
            if "real_model" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="requires --run-real-model"))


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    import os
    for key in list(os.environ):
        if key.startswith("RAG_WORKSHOP_") and not key.startswith("RAG_WORKSHOP_TEST_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("RAG_WORKSHOP_ENV", "test")
    monkeypatch.setenv("RAG_WORKSHOP_DATABASE_PATH", str(tmp_path / "db"))
    monkeypatch.setenv("RAG_WORKSHOP_STORAGE_ROOT", str(tmp_path / "objects"))
    monkeypatch.setenv("RAG_WORKSHOP_DOTENV", str(tmp_path / "absent.env"))
```

### `tests/unit/test_setup.py`

```python
from ragworkshop.cli import main
from ragworkshop.config import Settings


def test_info(capsys):
    assert main(["info"]) == 0
    assert capsys.readouterr().out == "ragworkshop: modular ingestion and retrieval\n"


def test_default_environment():
    assert Settings().environment == "development"
```

### `tests/integration/test_entrypoint.py`

```python
import subprocess


def test_installed_entrypoint(tmp_path):
    result = subprocess.run(["rag-workshop", "info"], cwd=tmp_path,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ragworkshop: modular ingestion and retrieval\n"
```

## Green — complete reference implementation

The exact filename precedes each complete file. Existing earlier-module files not shown here are unchanged. Empty package initializers and non-Python scaffolding are included so the checkpoint is reproducible.

The build backend discovers src packages and creates the executable. Runtime dependencies are declared once; dev adds tests/fixture generation and gguf remains optional. pytest importlib mode allows matching test filenames in separate directories.

### `pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "ragworkshop"
version = "0.1.0"
description = "Modular ingestion and retrieval workshop"
requires-python = ">=3.12"
dependencies = ["chromadb==1.5.9", "PyYAML==6.0.3", "python-dotenv==1.2.4", "python-docx==1.2.0", "pypdf==6.12.2"]

[project.optional-dependencies]
dev = ["pytest==9.2.0", "reportlab==4.4.10"]
gguf = ["llama-cpp-python==0.3.36"]

[project.scripts]
rag-workshop = "ragworkshop.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --import-mode=importlib"
markers = ["real_model: explicitly enabled local GGUF tests"]
```

This file supplies the declared package, configuration, or sample content used by this checkpoint.

### `.gitignore`

```text
.venv/
__pycache__/
.pytest_cache/
*.egg-info/
.env
.data/
models/
build/
dist/
```

This file supplies the declared package, configuration, or sample content used by this checkpoint.

### `.env.example`

```text
RAG_WORKSHOP_ENV=development
RAG_WORKSHOP_DATABASE_CONFIG=config/database.yml
RAG_WORKSHOP_STORAGE_CONFIG=config/storage.yml
# Opt-in local GGUF provider; default is the offline demo provider.
# RAG_WORKSHOP_EMBEDDING_PROVIDER=local
# RAG_WORKSHOP_EMBEDDING_MODEL=/absolute/path/to/embedding-model.gguf
# RAG_WORKSHOP_EMBEDDING_DIMENSIONS=768
```

This file supplies the declared package, configuration, or sample content used by this checkpoint.

### `config/database.yml`

```yaml
development:
  adapter: chroma
  path: ../.data/development/chroma
test:
  adapter: chroma
  path: ../.data/test/chroma
production:
  adapter: chroma
  path: ../.data/production/chroma
```

This file supplies the declared package, configuration, or sample content used by this checkpoint.

### `config/storage.yml`

```yaml
development:
  adapter: local
  root: ../.data/development/objects
test:
  adapter: local
  root: ../.data/test/objects
production:
  adapter: local
  root: ../.data/production/objects
```

### `README.md`

````markdown
# ragworkshop

Run commands from this directory after activating its virtual environment.

```bash
python -m pip install -e ".[dev]"
python -m pytest
rag-workshop info
```
````

This file supplies the declared package, configuration, or sample content used by this checkpoint.

### `src/ragworkshop/__init__.py`

```python
"""Modular RAG workshop package."""
```

This file supplies the declared package, configuration, or sample content used by this checkpoint.

### `src/ragworkshop/processors/__init__.py`

```python
"""Modular RAG workshop package."""
```

This file supplies the declared package, configuration, or sample content used by this checkpoint.

### `src/ragworkshop/embeddings/__init__.py`

```python
"""Modular RAG workshop package."""
```

This file supplies the declared package, configuration, or sample content used by this checkpoint.

### `src/ragworkshop/documents/__init__.py`

```python
"""Modular RAG workshop package."""
```

This file supplies the declared package, configuration, or sample content used by this checkpoint.

### `src/ragworkshop/storage/__init__.py`

```python
"""Modular RAG workshop package."""
```

This file supplies the declared package, configuration, or sample content used by this checkpoint.

### `src/ragworkshop/database/__init__.py`

```python
"""Modular RAG workshop package."""
```

This file supplies the declared package, configuration, or sample content used by this checkpoint.

### `src/ragworkshop/operations/__init__.py`

```python
"""Modular RAG workshop package."""
```

Settings is immutable. The resolver selects and validates environment sections, defines relative-path semantics, and reads dotenv only when explicitly called. Tests override paths before constructing adapters.

### `src/ragworkshop/config.py`

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    environment: str = "development"
```

ApplicationContext groups dependencies without hiding construction. create_context is a context manager that initializes migrations, enforces the embedding space, and respects provider ownership.

### `src/ragworkshop/context.py`

```python
"""Composition root is introduced in Module 5."""
```

main accepts argv for direct testing; argparse parses input and formats usage errors. JSON serialization is presentation logic. Command handlers delegate behavior and close providers they construct.

### `src/ragworkshop/cli.py`

```python
import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rag-workshop")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("info")
    parser.parse_args(argv)
    print("ragworkshop: modular ingestion and retrieval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

This file supplies the declared package, configuration, or sample content used by this checkpoint.

### `samples/sample.txt`

```text
Python modules make dependencies explicit.
```

## Refactor, pattern rationale, and alternatives

argparse keeps runtime dependencies small; Click or Typer could replace only the CLI adapter. A root package layout can work, but src layout makes installation mistakes easier to detect. Protocols versus ABCs will become a concrete choice when a replacement boundary appears.

RAG means retrieval-augmented generation. Ingestion brings documents into the system; extraction produces text; chunking splits it; embedding maps chunks to numeric vectors; indexing stores those vectors; retrieval compares a query vector to indexed vectors. Generation would supply retrieved text to a language model. **This course builds ingestion and retrieval, not a text-generation model.**

The CLI is an input adapter. Operations coordinate use cases. Domain records and interfaces describe what those operations need. Infrastructure adapters implement those interfaces. Dependencies point inward: a storage adapter may import the storage contract, but a chunker must not import a CLI or database client. Runtime calls can flow outward through an interface without reversing source-code dependencies.

SRP means a chunker changes for chunking reasons, not PDF parser changes. DIP means ingestion receives an embedding interface rather than constructing a particular model. OCP means another loader can be registered without modifying the ingestion algorithm. LSP means replacements preserve ordering, metadata, and failure contracts. ISP means the repository exposes only the operations this application actually needs, rather than a universal persistence framework.

Pattern preview: **Strategy** selects an interchangeable algorithm (format processors and embeddings); **Adapter** translates a library or filesystem API into an application interface; a **simple Factory** selects and constructs an implementation; **Repository** exposes domain persistence instead of Chroma dictionaries. These are ordinary Python objects and functions, not a framework. Dependency injection is passing those objects as arguments. A composition root is the single boundary that assembles them. Their cost is extra names and indirection; introduce them at actual replacement boundaries, not around every function.

A `src` layout keeps the repository root from accidentally making application code importable. Setuptools discovers `ragworkshop` under `src`; editable installation points the interpreter there. `__init__.py` identifies a regular package and should avoid environment reads and expensive initialization. Absolute imports begin with `ragworkshop`, never with the reference application's package. The reference uses a root-level application package; this workshop deliberately teaches installation rather than relying on the current directory.

Unit tests exercise small behavior with fakes; integration tests use a real boundary such as a parser or filesystem; E2E tests compose a complete use case. A subprocess test of `info` is entry-point integration, not a full ingestion E2E test. TDD turns each contract into an executable question before implementation. Red must fail for the intended missing behavior; Green is the smallest working behavior; Refactor changes structure while keeping assertions green.

The future FastAPI entry point would construct the same dependencies. Configuration belongs at entry points, not module import time. Neither a CLI nor an HTTP handler should contain chunking or database logic. YAML contains non-secret adapter/path choices; `.env` is local configuration, not something to commit.

## Verification and expected test outcomes

**Working directory: your `rag-workshop/` project root; activate `.venv` first.**

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m pytest -q
rag-workshop info
```

Validated cumulative result: **3 passed**.

Expected: all implemented tests pass; no model tests exist yet.

**Working directory: `rag-workshop/`.**

```bash
rag-workshop info
# ragworkshop: modular ingestion and retrieval
rag-workshop --help
# usage includes {info}
python -m ragworkshop.cli info
# ragworkshop: modular ingestion and retrieval
```

## Edge cases and limitations

An editable install does not make arbitrary directories part of a package. Missing console scripts usually mean the wrong virtual environment, a missing script table, or failure to reinstall after changing entry points. argparse should exit 2 for invalid commands; package imports should not create data directories.

## Solution architecture trace

Arrows show **installation followed by runtime dispatch**. This trace explains the completed behavior after the test-first implementation above.

```text
setuptools reads pyproject.toml
       |
       +--> finds ragworkshop under src/
       +--> registers rag-workshop -> ragworkshop.cli:main
                                             |
                                      parses argv ["info"]
                                             |
                  +--------------------------+----------------------+
                  |                                                 |
stdout: "ragworkshop: modular ingestion and retrieval"          return 0

unit: main(["info"]) verifies behavior
integration: installed executable verifies behavior + packaging
```

The integration test runs from a temporary working directory, so success cannot depend on the source directory being the current directory. `Settings` remains a separate, inert configuration scaffold. No model, storage adapter, or database is initialized merely to print application information.

## Why the acceptance criteria are met

- Create the directory tree, regular-package __init__.py files, virtual environment, and pyproject.toml. The completed code and assertions above make this behavior observable.
- Register the rag-workshop console script and src package discovery. The completed code and assertions above make this behavior observable.
- Write the provided unit and installed-entry-point assertions first. The completed code and assertions above make this behavior observable.
- Implement info and configuration scaffolding; install editable and run from a temporary directory. The completed code and assertions above make this behavior observable.

Tests are the executable acceptance evidence; successful imports alone are insufficient. Exact validation counts for all checkpoints are recorded in the workshop README. The optional real-model check is reported separately from default offline validation.
