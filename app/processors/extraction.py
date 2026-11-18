"""Dispatch document readers and translate parser errors at the file boundary."""
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO
from zipfile import BadZipFile
from struct import error as StructError

from lxml.etree import XMLSyntaxError
from openpyxl.utils.exceptions import InvalidFileException
from pypdf.errors import PyPdfError
from xlrd import XLRDError
from xlrd.compdoc import CompDocError

from app.exceptions import DocumentExtractionError, UnsupportedFileTypeError
from app.processors.extractors.pdf import extract as extract_pdf
from app.processors.extractors.word import extract as extract_docx
from app.processors.extractors.spreadsheet import extract_excel, extract_xls
from app.processors.extractors.presentation import extract as extract_pptx


EXTRACTORS: dict[str, Callable[[BinaryIO], str]] = {
    '.pdf': extract_pdf,
    '.docx': extract_docx,
    '.xlsx': extract_excel,
    '.xlsm': extract_excel,
    '.xls': extract_xls,
    '.pptx': extract_pptx,
}
SUPPORTED_EXTENSIONS = ('.txt', *EXTRACTORS)


def extract_file(path: Path) -> str:
    """Return text; preserve filesystem errors and sanitize parser diagnostics."""
    suffix = path.suffix.lower()
    if suffix == '.txt':
        try:
            return path.read_text(encoding='utf-8')
        except UnicodeError:
            raise DocumentExtractionError('Cannot extract .txt file: text must be valid UTF-8') from None
    extractor = EXTRACTORS.get(suffix)
    if extractor is None:
        raise UnsupportedFileTypeError('Unsupported file type; supported extensions: ' + ', '.join(SUPPORTED_EXTENSIONS))
    # Open outside parser error handling to preserve ordinary filesystem errors.
    with path.open('rb') as stream:
        try:
            return extractor(stream)
        except (BadZipFile, XMLSyntaxError, InvalidFileException, PyPdfError,
                XLRDError, CompDocError, StructError,
                ValueError, KeyError, IndexError, TypeError, EOFError):
            # Never expose raw parser diagnostics or document content.
            raise DocumentExtractionError(
                f'Cannot extract {suffix} file: document is malformed, encrypted, or unreadable'
            ) from None
