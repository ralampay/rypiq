"""Compatibility imports and entrypoints for spreadsheet readers."""
from typing import BinaryIO

from .excel import ExcelExtractor
from .xls import XlsExtractor


def extract_excel(stream: BinaryIO) -> str:
    return ExcelExtractor().extract(stream)


def extract_xls(stream: BinaryIO) -> str:
    return XlsExtractor().extract(stream)
