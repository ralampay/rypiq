import pytest

from app.processors.base import BaseProcessor
from app.processors.text_file import TextFileProcessor
from app.operations.text_file import process_text_file

@pytest.mark.unit
def test_processor_interface():
    processor = TextFileProcessor(
        chunk_size=4,
        stride=3,
    )

    assert isinstance(processor, BaseProcessor)
