"""Explicit selection of stateless format readers."""
from app.exceptions import UnsupportedFileTypeError
from app.processors.extractors.base import BaseExtractor
from app.processors.extractors.pdf import PdfExtractor
from app.processors.extractors.word import WordExtractor
from app.processors.extractors.excel import ExcelExtractor
from app.processors.extractors.xls import XlsExtractor
from app.processors.extractors.presentation import PresentationExtractor
from app.processors.extractors.text import TextExtractor


EXTRACTORS: dict[str, BaseExtractor] = {
    '.txt': TextExtractor(),
    '.pdf': PdfExtractor(),
    '.docx': WordExtractor(),
    '.xlsx': ExcelExtractor(),
    '.xlsm': ExcelExtractor(),
    '.xls': XlsExtractor(),
    '.pptx': PresentationExtractor(),
}
SUPPORTED_EXTENSIONS = tuple(EXTRACTORS)


class ExtractorFactory:
    def get(self, suffix: str) -> BaseExtractor:
        extractor = EXTRACTORS.get(suffix.lower())
        if extractor is None:
            raise UnsupportedFileTypeError(
                'Unsupported file type; supported extensions: ' + ', '.join(SUPPORTED_EXTENSIONS)
            )
        return extractor
