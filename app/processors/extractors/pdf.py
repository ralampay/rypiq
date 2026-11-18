"""Pdf text extraction from caller-owned binary streams."""
from typing import BinaryIO
from pypdf import PdfReader
from app.exceptions import ValidationError
from .base import BaseExtractor
from ._text import blocks


class PdfExtractor(BaseExtractor):
    def extract(self, stream: BinaryIO) -> str:
        reader = PdfReader(stream)
        if reader.is_encrypted:
            raise ValidationError('encrypted documents are not supported')
        return blocks(page.extract_text() or '' for page in reader.pages)


def extract(stream: BinaryIO) -> str:
    """Compatibility entrypoint for PdfExtractor."""
    return PdfExtractor().extract(stream)
