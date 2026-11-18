"""Presentation text extraction from caller-owned binary streams."""
from typing import BinaryIO
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE
from ._text import blocks, row


def _slide_shapes(shapes) -> str:
    values = []
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            values.append(_slide_shapes(shape.shapes))
        elif shape.has_table:
            values.extend(row(cell.text if not cell.is_spanned else '' for cell in table_row.cells)
                          for table_row in shape.table.rows)
        elif shape.has_text_frame:
            values.append(shape.text)
    return blocks(values)


def extract(stream: BinaryIO) -> str:
    return blocks(_slide_shapes(slide.shapes) for slide in Presentation(stream).slides)
