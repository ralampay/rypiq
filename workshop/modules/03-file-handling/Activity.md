# Activity 3: File handling

## Learning outcome

Generate small fixtures and test parser errors predictably.

## Problem and motivation

Colleagues have Word and text-based PDF documents. Extend the existing TXT loader without moving extraction logic into the chunker or adding embedding/database calls to readers.

## Prerequisites and starter setup

Complete Module 2 and keep its source and tests. Start from that checkpoint, never from the final application.

**Working directory: `~/rag-workshop/`, with `.venv` active.** `WORKSHOP_MATERIALS` is the absolute materials directory exported in Module 1. This overlay changes only this module's files; it preserves earlier completed exercises. It supplies infrastructure and replaces the exercise bodies with TODOs. Commit or copy your previous checkpoint first if you want to compare it later.

```bash
cp -R "$WORKSHOP_MATERIALS/starters/module-03/." .
python -m pip install -e ".[dev]"
```

Read the supplied scaffolding and write the tests below before filling any TODO. The overlay includes those completed test files as an instructor recovery aid; learners can retype or extend them before implementing behavior. Tests from previous modules remain regression checks.

**Files introduced or replaced in this module:** unchanged files from previous modules remain in place. A test may import a newly introduced TODO module and fail; that is intentional Red, not a dependency on a later module.

```text
rag-workshop/
  src/ragworkshop/cli.py
  src/ragworkshop/documents/models.py
  src/ragworkshop/documents/loaders.py
  src/ragworkshop/operations/documents.py
  tests/integration/test_documents.py
  tests/unit/test_documents.py
```

## Required tasks

1. Implement DocxFileLoader.load using body paragraphs with one-based paragraph numbers.
2. Implement PdfFileLoader.load using a context-managed stream and one-based page numbers; reject encrypted PDFs.
3. Implement get_loader using the extension registry; preserve the supplied TXT behavior.
4. Generate real TXT, DOCX, and PDF fixtures and complete the CLI extraction checks.
5. Keep chunking in the shared operation and retain source, filename, segment index, and provenance.

## Exact exercise files and interface stubs

These are the files to create or replace. Bodies marked TODO are deliberately incomplete. All other changed files in the tree are supplied scaffolding, not hidden requirements. The stage overlay contains exactly these stubs and the tests below. Keep public signatures unchanged.

### `src/ragworkshop/documents/loaders.py`

```python
from pathlib import Path
from typing import Protocol
from ragworkshop.documents.models import ExtractedDocument, TextSegment

class DocumentLoader(Protocol):

    def load(self, path: Path) -> ExtractedDocument:
        ...

def validate_path(path: Path, extension: str) -> Path:
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError('Input must be a regular file')
    if path.suffix.lower() != extension:
        raise ValueError(f'Expected {extension} file')
    return path

class TextFileLoader:

    def load(self, path: Path) -> ExtractedDocument:
        path = validate_path(path, '.txt')
        return ExtractedDocument(path, [TextSegment(path.read_text(encoding='utf-8'), {'segment': 1})])

class DocxFileLoader:

    def load(self, path: Path) -> ExtractedDocument:
        raise NotImplementedError('TODO: DocxFileLoader.load')

class PdfFileLoader:

    def load(self, path: Path) -> ExtractedDocument:
        raise NotImplementedError('TODO: PdfFileLoader.load')
LOADERS: dict[str, type[DocumentLoader]] = {'.txt': TextFileLoader, '.docx': DocxFileLoader, '.pdf': PdfFileLoader}

def get_loader(path: Path) -> DocumentLoader:
    raise NotImplementedError('TODO: get_loader')
```

## Red — tests to write or complete

Write these tests before implementing the TODOs. You may use the supplied test files as a starting point; add one edge-case assertion of your own. Do not weaken assertions or mark unfinished behavior skipped.

### `tests/integration/test_documents.py`

```python
from pathlib import Path
import subprocess
import json

import pytest
from docx import Document
from reportlab.pdfgen.canvas import Canvas

from ragworkshop.documents.loaders import get_loader
from ragworkshop.operations.documents import process_document
from ragworkshop.processors.chunks import TextChunker


@pytest.fixture
def documents(tmp_path):
    txt = tmp_path / "lesson.txt"
    txt.write_text("python modules", encoding="utf-8")
    docx = tmp_path / "lesson.docx"
    document = Document()
    document.add_paragraph("python modules")
    document.add_paragraph("database storage")
    document.save(docx)
    pdf = tmp_path / "lesson.pdf"
    canvas = Canvas(str(pdf), invariant=1)
    canvas.drawString(72, 720, "python modules")
    canvas.showPage()
    canvas.drawString(72, 720, "database storage")
    canvas.save()
    return [txt, docx, pdf]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_real_parsers(documents, index):
    path = documents[index]
    chunks = process_document(path, TextChunker(100, 100))
    assert "python modules" in chunks[0].text
    assert chunks[0].metadata["source"] == str(path)
    assert chunks[0].start == 0
    if index == 1:
        assert chunks[1].metadata["paragraph"] == 2
    if index == 2:
        assert chunks[1].metadata["page"] == 2
    result = subprocess.run(["rag-workshop", "process-file", str(path)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "python modules" in json.loads(result.stdout)["chunks"][0]["text"]


def test_missing_unsupported_and_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_loader(tmp_path / "absent.txt").load(tmp_path / "absent.txt")
    with pytest.raises(ValueError, match="Unsupported"):
        get_loader(tmp_path / "file.exe")
    path = tmp_path / "empty.txt"
    path.write_text(" ", encoding="utf-8")
    assert process_document(path, TextChunker()) == []


@pytest.mark.parametrize("suffix", [".pdf", ".docx"])
def test_corrupt_file(tmp_path, suffix):
    path = tmp_path / ("bad" + suffix)
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="Cannot parse"):
        get_loader(path).load(path)


def test_unreadable(monkeypatch, tmp_path):
    path = tmp_path / "locked.txt"
    path.touch()
    def denied(self, **kwargs):
        raise PermissionError("denied")
    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(PermissionError):
        get_loader(path).load(path)
```

### `tests/unit/test_documents.py`

```python
from pathlib import Path
from ragworkshop.documents.models import ExtractedDocument, TextSegment
from ragworkshop.operations import documents
from ragworkshop.processors.chunks import TextChunker


def test_chunking_resets_offsets_but_not_indexes(monkeypatch):
    class Loader:
        def load(self, path):
            return ExtractedDocument(Path("memo.txt"), [
                TextSegment("abcd", {"paragraph": 1}), TextSegment("ef", {"paragraph": 2})])
    monkeypatch.setattr(documents, "get_loader", lambda path: Loader())
    chunks = documents.process_document(Path("memo.txt"), TextChunker(3, 3))
    assert [c.index for c in chunks] == [0, 1, 2]
    assert [c.start for c in chunks] == [0, 3, 0]
    assert chunks[-1].metadata["paragraph"] == 2
```

## Green and Refactor constraints

- Implement DocxFileLoader.load using body paragraphs with one-based paragraph numbers.
- Implement PdfFileLoader.load using a context-managed stream and one-based page numbers; reject encrypted PDFs.
- Implement get_loader using the extension registry; preserve the supplied TXT behavior.
- Generate real TXT, DOCX, and PDF fixtures and complete the CLI extraction checks.
- Keep chunking in the shared operation and retain source, filename, segment index, and provenance.
- First satisfy one failing assertion, then rerun the current module tests.
- Refactor only after Green; preserve earlier commands and tests.
- Keep business behavior out of CLI branches. Use supplied interfaces and explicit arguments.
- Keep automated tests offline, with temporary paths for file/database interactions.
- No GPU, cloud credentials, or real GGUF is required.

## Input/output examples and expected observable behavior

**Working directory: `rag-workshop/`.** Output comments specify assertions, not byte-for-byte JSON formatting.

```bash
rag-workshop process-file samples/sample.txt
# one chunk; filename sample.txt; segment_index 0; source is an absolute path
rag-workshop process-file samples/lesson.docx
# two chunks with paragraph values 1 and 2
rag-workshop process-file samples/lesson.pdf
# two chunks with page values 1 and 2
```

**Working directory: `rag-workshop/`.** Create the manual demonstration fixtures after installing the dev extra. The same generation approach is tested automatically in temporary directories.

```bash
python - <<'PYFIX'
from pathlib import Path
from docx import Document
from reportlab.pdfgen.canvas import Canvas

Path("samples").mkdir(exist_ok=True)
document = Document()
document.add_paragraph("python modules")
document.add_paragraph("database storage")
document.save("samples/lesson.docx")
canvas = Canvas("samples/lesson.pdf", invariant=1)
canvas.drawString(72, 720, "python modules")
canvas.showPage()
canvas.drawString(72, 720, "database storage")
canvas.save()
PYFIX
```

## Acceptance criteria

- Implement DocxFileLoader.load using body paragraphs with one-based paragraph numbers.
- Implement PdfFileLoader.load using a context-managed stream and one-based page numbers; reject encrypted PDFs.
- Implement get_loader using the extension registry; preserve the supplied TXT behavior.
- Generate real TXT, DOCX, and PDF fixtures and complete the CLI extraction checks.
- Keep chunking in the shared operation and retain source, filename, segment index, and provenance.
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

Validated cumulative result: **40 passed, 1 skipped**.

Expected: all implemented tests pass; the real-GGUF test is skipped unless explicitly opted in. This skip is expected, not evidence that real inference was tested.

## Optional stretch challenges

- Add DOCX table extraction with explicit table/row/cell provenance.
- Add a test explaining multi-column PDF extraction order; do not claim OCR support.
