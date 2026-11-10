# Activity 1: Introduction and project setup

## Learning outcome

Run a failing test, implement the smallest change, and refactor safely.

## Problem and motivation

Your team needs a tool that behaves the same outside an IDE and outside its source directory. Package a minimal CLI so installation and import boundaries are verified before adding RAG behavior.

## Prerequisites and starter setup

Basic Python functions/classes/imports; Linux shell and Python 3.12+ with venv.

**Working directory initially: the repository root containing `workshop/`.** Keep the teaching source and your student project separate; no reference application file needs to change.

```bash
export WORKSHOP_MATERIALS="$PWD/workshop"
mkdir -p "$HOME/rag-workshop"
cd "$HOME/rag-workshop"
python3 --version
# Python 3.12 or newer is required; the delivered examples were tested with 3.14.3.
python3 -m venv .venv
source .venv/bin/activate
mkdir -p config samples tests/{unit,integration,e2e}
mkdir -p src/ragworkshop/{processors,embeddings,documents,storage,database,operations}
cp -R "$WORKSHOP_MATERIALS/starters/module-01/." .
```

The starter creates exact files listed below, including regular-package `__init__.py` files. Inspect and complete the two TODO settings in `pyproject.toml` (script registration and `where = ["src"]`) before installing. Use the provided dependency declarations; installing a native model runtime is not required. On distributions without venv, install the distribution's `python3-venv` package before class.

**Working directory now and for subsequent module commands: `~/rag-workshop/`.**

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/unit/test_setup.py -q
# Red: main still raises NotImplementedError.
```

An installation needs package access or a pre-populated wheel cache; the default test suite itself needs neither network access nor model files. Install dependencies before the class when possible. Reinstall after changing entry points; ordinary editable Python changes are visible immediately.

**Files introduced or replaced in this module:** unchanged files from previous modules remain in place. A test may import a newly introduced TODO module and fail; that is intentional Red, not a dependency on a later module.

```text
rag-workshop/
  pyproject.toml
  .gitignore
  .env.example
  config/database.yml
  config/storage.yml
  README.md
  src/ragworkshop/__init__.py
  src/ragworkshop/processors/__init__.py
  src/ragworkshop/embeddings/__init__.py
  src/ragworkshop/documents/__init__.py
  src/ragworkshop/storage/__init__.py
  src/ragworkshop/database/__init__.py
  src/ragworkshop/operations/__init__.py
  src/ragworkshop/config.py
  src/ragworkshop/context.py
  src/ragworkshop/cli.py
  tests/conftest.py
  tests/unit/test_setup.py
  tests/integration/test_entrypoint.py
  samples/sample.txt
```

## Required tasks

1. Create the directory tree, regular-package __init__.py files, virtual environment, and pyproject.toml.
2. Register the rag-workshop console script and src package discovery.
3. Write the provided unit and installed-entry-point assertions first.
4. Implement info and configuration scaffolding; install editable and run from a temporary directory.

## Exact exercise files and interface stubs

These are the files to create or replace. Bodies marked TODO are deliberately incomplete. All other changed files in the tree are supplied scaffolding, not hidden requirements. The stage overlay contains exactly these stubs and the tests below. Keep public signatures unchanged.

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
# TODO: register rag-workshop = "ragworkshop.cli:main"

[tool.setuptools.packages.find]
# TODO: set where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --import-mode=importlib"
markers = ["real_model: explicitly enabled local GGUF tests"]
```

### `src/ragworkshop/cli.py`

```python
import argparse

def main(argv: list[str] | None=None) -> int:
    raise NotImplementedError('TODO: main')
if __name__ == '__main__':
    raise SystemExit(main())
```

## Red — tests to write or complete

Write these tests before implementing the TODOs. You may use the supplied test files as a starting point; add one edge-case assertion of your own. Do not weaken assertions or mark unfinished behavior skipped.

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

## Green and Refactor constraints

- Create the directory tree, regular-package __init__.py files, virtual environment, and pyproject.toml.
- Register the rag-workshop console script and src package discovery.
- Write the provided unit and installed-entry-point assertions first.
- Implement info and configuration scaffolding; install editable and run from a temporary directory.
- First satisfy one failing assertion, then rerun the current module tests.
- Refactor only after Green; preserve earlier commands and tests.
- Keep business behavior out of CLI branches. Use supplied interfaces and explicit arguments.
- Keep automated tests offline, with temporary paths for file/database interactions.
- No GPU, cloud credentials, or real GGUF is required.

## Input/output examples and expected observable behavior

**Working directory: `rag-workshop/`.** Output comments specify assertions, not byte-for-byte JSON formatting.

```bash
rag-workshop info
# ragworkshop: modular ingestion and retrieval
rag-workshop --help
# usage includes {info}
python -m ragworkshop.cli info
# ragworkshop: modular ingestion and retrieval
```

## Acceptance criteria

- Create the directory tree, regular-package __init__.py files, virtual environment, and pyproject.toml.
- Register the rag-workshop console script and src package discovery.
- Write the provided unit and installed-entry-point assertions first.
- Implement info and configuration scaffolding; install editable and run from a temporary directory.
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

Validated cumulative result: **3 passed**.

Expected: all implemented tests pass; no model tests exist yet.

## Optional stretch challenges

- Add a `--version` option with an installed-entry-point test. Explain whether version data belongs in the CLI or package metadata.
