"""Dispatch document readers and translate parser errors at the file boundary."""
from pathlib import Path
from zipfile import BadZipFile
from struct import error as StructError

from lxml.etree import XMLSyntaxError
from openpyxl.utils.exceptions import InvalidFileException
from pypdf.errors import PyPdfError
from xlrd import XLRDError
from xlrd.compdoc import CompDocError

from app.exceptions import DocumentExtractionError
from app.processors.extractors.factory import ExtractorFactory


class FileTextReader:
    def __init__(self, *, factory: ExtractorFactory | None = None) -> None:
        self.factory = factory if factory is not None else ExtractorFactory()

    def read(self, path: Path) -> str:
        """Own the file stream; preserve filesystem errors and sanitize parser errors."""
        suffix = path.suffix.lower()
        extractor = self.factory.get(suffix)
        # Open outside parser error handling to preserve ordinary filesystem errors.
        with path.open('rb') as stream:
            try:
                return extractor.extract(stream)
            except (BadZipFile, XMLSyntaxError, InvalidFileException, PyPdfError,
                    XLRDError, CompDocError, StructError,
                    ValueError, KeyError, IndexError, TypeError, EOFError) as error:
                if suffix == '.txt' and isinstance(error, UnicodeError):
                    raise DocumentExtractionError('Cannot extract .txt file: text must be valid UTF-8') from None
                # Never expose raw parser diagnostics or document content.
                raise DocumentExtractionError(
                    f'Cannot extract {suffix} file: document is malformed, encrypted, or unreadable'
                ) from None
