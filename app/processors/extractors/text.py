"""UTF-8 text extraction with universal newline handling."""
from io import TextIOWrapper
from typing import BinaryIO

from .base import BaseExtractor


class TextExtractor(BaseExtractor):
    def extract(self, stream: BinaryIO) -> str:
        reader = TextIOWrapper(stream, encoding='utf-8', newline=None)
        try:
            return reader.read()
        finally:
            # The caller owns the binary stream, including on decoding failure.
            reader.detach()
