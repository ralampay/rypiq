"""Word text extraction from caller-owned binary streams."""
from typing import BinaryIO
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from .base import BaseExtractor
from ._text import blocks, row


def _word_blocks(container) -> str:
    values = []
    for item in container.iter_inner_content():
        if isinstance(item, Paragraph):
            values.append(item.text)
        elif isinstance(item, Table):
            for table_row in item.rows:
                # A merged cell can appear repeatedly in python-docx's grid.
                seen = set()
                cells = []
                for cell in table_row.cells:
                    if cell._tc in seen:
                        cells.append('')
                    else:
                        seen.add(cell._tc)
                        cells.append(_word_blocks(cell))
                values.append(row(cells))
    return blocks(values)


class WordExtractor(BaseExtractor):
    def extract(self, stream: BinaryIO) -> str:
        return _word_blocks(Document(stream))


def extract(stream: BinaryIO) -> str:
    """Compatibility entrypoint for WordExtractor."""
    return WordExtractor().extract(stream)
