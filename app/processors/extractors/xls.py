"""Legacy Excel text extraction from a borrowed binary stream."""
from io import StringIO
from typing import BinaryIO
import xlrd

from .base import BaseExtractor
from ._spreadsheet import worksheet
from ._text import blocks


def _xls_cell(cell, datemode: int) -> object:
    if cell.ctype == xlrd.XL_CELL_DATE:
        return xlrd.xldate_as_datetime(cell.value, datemode)
    if cell.ctype == xlrd.XL_CELL_BOOLEAN:
        return bool(cell.value)
    if cell.ctype == xlrd.XL_CELL_ERROR:
        return xlrd.error_text_from_code.get(cell.value, '#ERROR!')
    return cell.value


class XlsExtractor(BaseExtractor):
    def extract(self, stream: BinaryIO) -> str:
        book = xlrd.open_workbook(file_contents=stream.read(), on_demand=True, logfile=StringIO())
        try:
            return blocks(worksheet(sheet.name, (
                (_xls_cell(cell, book.datemode) for cell in sheet.row(index))
                for index in range(sheet.nrows)
            )) for sheet in book.sheets())
        finally:
            book.release_resources()

