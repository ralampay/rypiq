"""Extract document text without selecting models or changing source files."""
from collections.abc import Callable, Iterable
from datetime import date, datetime, time
from io import StringIO
from pathlib import Path
from typing import BinaryIO
from zipfile import BadZipFile
from struct import error as StructError

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from lxml.etree import XMLSyntaxError
from openpyxl import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from pypdf import PdfReader
from pypdf.errors import PyPdfError
import xlrd
from xlrd.compdoc import CompDocError


def _blocks(values: Iterable[str]) -> str:
    return '\n'.join(value for value in values if value.strip())


def _cell(value: object) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _row(values: Iterable[object]) -> str:
    cells = [_cell(value) for value in values]
    while cells and not cells[-1].strip():
        cells.pop()
    return '\t'.join(cells)


def _pdf(stream: BinaryIO) -> str:
    reader = PdfReader(stream)
    if reader.is_encrypted:
        raise ValueError('encrypted documents are not supported')
    return _blocks(page.extract_text() or '' for page in reader.pages)


def _word_blocks(container) -> str:
    blocks = []
    for item in container.iter_inner_content():
        if isinstance(item, Paragraph):
            blocks.append(item.text)
        elif isinstance(item, Table):
            for row in item.rows:
                # A merged cell can appear repeatedly in python-docx's grid.
                seen = set()
                cells = []
                for cell in row.cells:
                    if cell._tc in seen:
                        cells.append('')
                    else:
                        seen.add(cell._tc)
                        cells.append(_word_blocks(cell))
                blocks.append(_row(cells))
    return _blocks(blocks)


def _docx(stream: BinaryIO) -> str:
    return _word_blocks(Document(stream))


def _worksheet(name: str, rows: Iterable[Iterable[object]]) -> str:
    content = _blocks(_row(row) for row in rows)
    return name + '\n' + content if content else ''


def _excel(stream: BinaryIO) -> str:
    book = load_workbook(stream, read_only=True, data_only=True, keep_links=False)
    try:
        return _blocks(_worksheet(sheet.title, sheet.iter_rows(values_only=True))
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


def _xls(stream: BinaryIO) -> str:
    book = xlrd.open_workbook(file_contents=stream.read(), on_demand=True, logfile=StringIO())
    try:
        return _blocks(_worksheet(sheet.name, (
            (_xls_cell(cell, book.datemode) for cell in sheet.row(index))
            for index in range(sheet.nrows)
        )) for sheet in book.sheets())
    finally:
        book.release_resources()


def _slide_shapes(shapes) -> str:
    blocks = []
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            blocks.append(_slide_shapes(shape.shapes))
        elif shape.has_table:
            blocks.extend(_row(cell.text if not cell.is_spanned else '' for cell in row.cells)
                          for row in shape.table.rows)
        elif shape.has_text_frame:
            blocks.append(shape.text)
    return _blocks(blocks)


def _pptx(stream: BinaryIO) -> str:
    return _blocks(_slide_shapes(slide.shapes) for slide in Presentation(stream).slides)


EXTRACTORS: dict[str, Callable[[BinaryIO], str]] = {
    '.pdf': _pdf,
    '.docx': _docx,
    '.xlsx': _excel,
    '.xlsm': _excel,
    '.xls': _xls,
    '.pptx': _pptx,
}
SUPPORTED_EXTENSIONS = ('.txt', *EXTRACTORS)


def extract_file(path: Path) -> str:
    """Return text; preserve filesystem errors and sanitize parser diagnostics."""
    suffix = path.suffix.lower()
    if suffix == '.txt':
        return path.read_text(encoding='utf-8')
    extractor = EXTRACTORS.get(suffix)
    if extractor is None:
        raise ValueError('Unsupported file type; supported extensions: ' + ', '.join(SUPPORTED_EXTENSIONS))
    # Open outside parser error handling to preserve ordinary filesystem errors.
    with path.open('rb') as stream:
        try:
            return extractor(stream)
        except (BadZipFile, XMLSyntaxError, InvalidFileException, PyPdfError,
                xlrd.XLRDError, CompDocError, StructError,
                ValueError, KeyError, IndexError, TypeError, EOFError):
            # Never expose raw parser diagnostics or document content.
            raise ValueError(
                f'Cannot extract {suffix} file: document is malformed, encrypted, or unreadable'
            ) from None
