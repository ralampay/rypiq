"""Modern Excel text extraction from a borrowed binary stream."""
from typing import BinaryIO
from openpyxl import load_workbook

from .base import BaseExtractor
from ._spreadsheet import worksheet
from ._text import blocks


class ExcelExtractor(BaseExtractor):
    def extract(self, stream: BinaryIO) -> str:
        book = load_workbook(stream, read_only=True, data_only=True, keep_links=False)
        try:
            return blocks(worksheet(sheet.title, sheet.iter_rows(values_only=True))
                           for sheet in book.worksheets)
        finally:
            book.close()

