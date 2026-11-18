"""Compatibility entrypoint and registry exports for file text extraction."""
from pathlib import Path

from app.processors.extractors.factory import EXTRACTORS, SUPPORTED_EXTENSIONS
from app.processors.file_reader import FileTextReader


def extract_file(path: Path) -> str:
    """Return extracted text using an owned file stream."""
    return FileTextReader().read(path)
