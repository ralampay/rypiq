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
