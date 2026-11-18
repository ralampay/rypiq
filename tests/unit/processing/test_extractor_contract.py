"""Shared extractor interface and borrowed text-stream lifetime."""
from io import BytesIO

import pytest

from app.processors.extraction import EXTRACTORS, SUPPORTED_EXTENSIONS
from app.processors.extractors.base import BaseExtractor
from app.processors.extractors.text import TextExtractor


def test_abstract_contract_requires_extract():
    class IncompleteExtractor(BaseExtractor):
        pass

    for cls in (BaseExtractor, IncompleteExtractor):
        with pytest.raises(TypeError, match='abstract'):
            cls()


def test_all_formats_use_the_extractor_contract():
    assert set(EXTRACTORS) == {'.txt', '.pdf', '.docx', '.xlsx', '.xlsm', '.xls', '.pptx'}
    assert SUPPORTED_EXTENSIONS == tuple(EXTRACTORS)
    assert all(isinstance(reader, BaseExtractor) for reader in EXTRACTORS.values())


@pytest.mark.parametrize('content', [b'hello\r\nworld\r', b'\xff'])
def test_text_reader_leaves_borrowed_stream_open(content):
    with BytesIO(content) as stream:
        if content == b'\xff':
            with pytest.raises(UnicodeError):
                TextExtractor().extract(stream)
        else:
            assert TextExtractor().extract(stream) == 'hello\nworld\n'
        assert not stream.closed
        stream.seek(0)
        assert stream.read() == content
