# Activity 3 solution: File handling

**Use this file during review or recovery.**

## Match to the exercise

Colleagues have Word and text-based PDF documents. Extend the existing TXT loader without moving extraction logic into the chunker or adding embedding/database calls to readers.

This solution uses exactly the activity signatures and requirements. It is a cumulative change: retain the previous module files and replace/add only the files listed here. The direct link below opens the complete cumulative checkpoint for this module.

[Complete Module 3 checkpoint](../../reference/module-03/)

## Red — completed tests first

Run these against the activity stubs and confirm the TODO failures before implementing. These are the same test contracts as Activity.md, with no changed expectations.

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
from ragworkshop.operations.documents import process_document

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
    command = commands.add_parser("process-file")
    command.add_argument("path", type=Path)
    command.add_argument("--chunk-size", type=int, default=200)
    command.add_argument("--stride", type=int, default=150)
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
        elif args.command == "process-file":
            chunks = process_document(args.path, TextChunker(args.chunk_size, args.stride))
            result = {"chunks": [asdict(chunk) for chunk in chunks]}
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

### `src/ragworkshop/documents/models.py`

```python
from dataclasses import dataclass, field
from pathlib import Path
from ragworkshop.processors.chunks import Metadata


@dataclass(frozen=True)
class TextSegment:
    text: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedDocument:
    source: Path
    segments: list[TextSegment]
```

Readers validate paths and translate library results into segments. The registry chooses a reader; readers never embed, index, or chunk.

### `src/ragworkshop/documents/loaders.py`

```python
from pathlib import Path
from typing import Protocol

from ragworkshop.documents.models import ExtractedDocument, TextSegment


class DocumentLoader(Protocol):
    def load(self, path: Path) -> ExtractedDocument: ...


def validate_path(path: Path, extension: str) -> Path:
    path = Path(path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("Input must be a regular file")
    if path.suffix.lower() != extension:
        raise ValueError(f"Expected {extension} file")
    return path


class TextFileLoader:
    def load(self, path: Path) -> ExtractedDocument:
        path = validate_path(path, ".txt")
        return ExtractedDocument(path, [TextSegment(path.read_text(encoding="utf-8"), {"segment": 1})])


class DocxFileLoader:
    def load(self, path: Path) -> ExtractedDocument:
        from docx import Document
        path = validate_path(path, ".docx")
        try:
            document = Document(path)
            segments = [TextSegment(paragraph.text, {"paragraph": i})
                        for i, paragraph in enumerate(document.paragraphs, 1)]
        except PermissionError:
            raise
        except Exception as exc:
            raise ValueError(f"Cannot parse DOCX: {path.name}") from exc
        return ExtractedDocument(path, segments)


class PdfFileLoader:
    def load(self, path: Path) -> ExtractedDocument:
        from pypdf import PdfReader
        path = validate_path(path, ".pdf")
        try:
            with path.open("rb") as stream:
                reader = PdfReader(stream)
                if reader.is_encrypted:
                    raise ValueError("Encrypted PDFs are not supported")
                segments = [TextSegment(page.extract_text() or "", {"page": i})
                            for i, page in enumerate(reader.pages, 1)]
        except PermissionError:
            raise
        except Exception as exc:
            raise ValueError(f"Cannot parse PDF: {path.name}") from exc
        return ExtractedDocument(path, segments)


LOADERS: dict[str, type[DocumentLoader]] = {
    ".txt": TextFileLoader, ".docx": DocxFileLoader, ".pdf": PdfFileLoader,
}


def get_loader(path: Path) -> DocumentLoader:
    try:
        loader = LOADERS[Path(path).suffix.lower()]
    except KeyError as exc:
        raise ValueError("Unsupported file format") from exc
    return loader()
```

The operation combines segments with source metadata, delegates chunking, skips whitespace-only chunks, and assigns document-wide indexes.

### `src/ragworkshop/operations/documents.py`

```python
from dataclasses import replace
from pathlib import Path

from ragworkshop.documents.loaders import get_loader
from ragworkshop.processors.chunks import TextChunk, TextChunker


def process_document(path: Path, chunker: TextChunker) -> list[TextChunk]:
    document = get_loader(path).load(path)
    chunks = []
    for segment_index, segment in enumerate(document.segments):
        metadata = {**segment.metadata, "source": str(document.source),
                    "filename": document.source.name, "segment_index": segment_index}
        for chunk in chunker.chunk(segment.text, metadata):
            if chunk.text.strip():
                chunks.append(replace(chunk, index=len(chunks)))
    return chunks
```

## Refactor, pattern rationale, and alternatives

A loader registry is easier to extend than an if/elif branch in the CLI. For very few stable formats a factory conditional is acceptable. Whole-document flattening is shorter but loses reliable page/paragraph attribution; segment-based extraction makes that tradeoff explicit.

The filesystem is an input boundary, not a text representation. A `.json` string passed to `process` is already text; a `.docx` path needs a parser. The loader Strategy chooses a file reader by case-insensitive extension. Each concrete loader is also an Adapter: it translates `python-docx` or `pypdf` objects into `ExtractedDocument` and `TextSegment`. The registry owns selection; the operation owns sequencing; the chunker owns window rules.

Extension checking is a routing and error-reporting convenience, not proof of a file's contents. Resolve the supplied path, require a regular file, and let the parser validate the format. Never execute document content. Missing files, permission errors, corrupt formats, and unsupported extensions have distinct causes. Parser exceptions become readable format errors with the original cause retained.

The document is a sequence of text segments. TXT has one segment, DOCX has body paragraphs, PDF has pages. Chunking restarts offsets at each segment; chunk indexes remain unique across the document. Source path, filename, segment index, and paragraph/page number travel with each chunk. Empty segments contribute no chunks, but provenance numbering is not compacted: paragraph 3 remains paragraph 3 even when paragraph 2 was blank.

`python-docx` body paragraphs omit tables, headers, text boxes, and some tracked changes. This is a stated extraction scope, not complete Word document fidelity. PDF page order follows the file, but within-page reading order can be surprising for columns and positioned text. `pypdf` extracts text; it does not OCR scanned images. A scanned-only document yields no usable chunks, and ingestion later rejects it with an OCR hint. Encrypted PDFs are intentionally rejected.

Integration fixtures must contain actual text. A blank `PdfWriter` page cannot prove extraction. ReportLab writes a tiny, deterministic text PDF (`invariant=1`); `python-docx` writes two simple paragraphs. Assertions compare text and provenance rather than DOCX ZIP timestamps or binary equality between separate fixture generations. A permission-denied test injects `PermissionError`; chmod-based tests can behave differently when run as root.

## Verification and expected test outcomes

**Working directory: your `rag-workshop/` project root; activate `.venv` first.**

```bash
python -m pytest tests/unit -q
python -m pytest tests/integration -q
python -m pytest -q
rag-workshop info
```

Validated cumulative result: **40 passed, 1 skipped**.

Expected: all implemented tests pass; the real-GGUF test is skipped unless explicitly opted in. This skip is expected, not evidence that real inference was tested.

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

**Working directory: `rag-workshop/`.**

```bash
rag-workshop process-file samples/sample.txt
# one chunk; filename sample.txt; segment_index 0; source is an absolute path
rag-workshop process-file samples/lesson.docx
# two chunks with paragraph values 1 and 2
rag-workshop process-file samples/lesson.pdf
# two chunks with page values 1 and 2
```

## Edge cases and limitations

Empty extraction produces no chunks; ingestion later rejects it. Missing files and permission errors are not successful empty documents. A valid suffix with invalid bytes must fail. DOCX tables and PDF OCR remain outside the declared extraction scope; generated fixtures exercise supported content only.

## Why the acceptance criteria are met

- Implement DocxFileLoader.load using body paragraphs with one-based paragraph numbers. The completed code and assertions above make this behavior observable.
- Implement PdfFileLoader.load using a context-managed stream and one-based page numbers; reject encrypted PDFs. The completed code and assertions above make this behavior observable.
- Implement get_loader using the extension registry; preserve the supplied TXT behavior. The completed code and assertions above make this behavior observable.
- Generate real TXT, DOCX, and PDF fixtures and complete the CLI extraction checks. The completed code and assertions above make this behavior observable.
- Keep chunking in the shared operation and retain source, filename, segment index, and provenance. The completed code and assertions above make this behavior observable.

Tests are the executable acceptance evidence; successful imports alone are insufficient. Exact validation counts for all checkpoints are recorded in the workshop README. The optional real-model check is reported separately from default offline validation.
