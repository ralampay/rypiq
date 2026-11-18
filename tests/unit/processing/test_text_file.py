import pytest

from app.processors.base import BaseProcessor
from app.processors.text_file import TextFileProcessor
from app.operations.processing.file import process_text_file

@pytest.mark.unit
def test_processor_interface():
    processor = TextFileProcessor(
        chunk_size=4,
        stride=3,
    )

    assert isinstance(processor, BaseProcessor)


def test_processor_uses_injected_reader_and_chunker():
    from pathlib import Path
    from unittest.mock import Mock

    reader = Mock()
    reader.read.return_value = 'sample text'
    chunker = Mock()
    chunks = [{'index': 0, 'start': 0, 'end': 6, 'text': 'sample'}]
    chunker.chunk.return_value = chunks
    processor = TextFileProcessor(6, 4, reader=reader, chunker=chunker)
    reader.read.assert_not_called()
    chunker.chunk.assert_not_called()

    assert processor.process('sample.pdf') == {
        'medium': 'text-file', 'source': 'sample.pdf', 'chunk_size': 6,
        'stride': 4, 'total_characters': 11, 'chunk_count': 1, 'chunks': chunks,
    }
    reader.read.assert_called_once_with(Path('sample.pdf'))
    chunker.chunk.assert_called_once_with('sample text', chunk_size=6, stride=4)


def test_reader_failure_does_not_invoke_chunker():
    from unittest.mock import Mock

    failure = OSError('source unavailable')
    reader = Mock()
    reader.read.side_effect = failure
    chunker = Mock()
    processor = TextFileProcessor(6, 4, reader=reader, chunker=chunker)
    with pytest.raises(OSError) as caught:
        processor.process('sample.pdf')
    assert caught.value is failure
    chunker.chunk.assert_not_called()
