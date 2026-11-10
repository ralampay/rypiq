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
