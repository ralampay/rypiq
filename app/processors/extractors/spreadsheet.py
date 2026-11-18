"""Spreadsheet text extraction from caller-owned binary streams."""
from typing import BinaryIO
from collections.abc import Iterable
from io import StringIO
from openpyxl import load_workbook
import xlrd
from ._text import blocks, row


def _worksheet(name: str, rows: Iterable[Iterable[object]]) -> str:
    content = blocks(row(values) for values in rows)
    return name + '\n' + content if content else ''


def extract_excel(stream: BinaryIO) -> str:
    book = load_workbook(stream, read_only=True, data_only=True, keep_links=False)
    try:
        return blocks(_worksheet(sheet.title, sheet.iter_rows(values_only=True))
                       for sheet in book.worksheets)
    finally:
        book.close()


def _xls_cell(cell, datemode: int) -> object:
    if cell.ctype == xlrd.XL_CELL_DATE:
        return xlrd.xldate_as_datetime(cell.value, datemode)
    if cell.ctype == xlrd.XL_CELL_BOOLEAN:
        return bool(cell.value)
    if cell.ctype == xlrd.XL_CELL_ERROR:
        return xlrd.error_text_from_code.get(cell.value, '#ERROR!')
    return cell.value


def extract_xls(stream: BinaryIO) -> str:
    book = xlrd.open_workbook(file_contents=stream.read(), on_demand=True, logfile=StringIO())
    try:
        return blocks(_worksheet(sheet.name, (
            (_xls_cell(cell, book.datemode) for cell in sheet.row(index))
            for index in range(sheet.nrows)
        )) for sheet in book.sheets())
    finally:
        book.release_resources()
