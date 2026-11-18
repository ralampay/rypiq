"""Character chunking independently of file and model resources."""
import pytest

from app.exceptions import ValidationError
from app.processors.chunkers.base import BaseChunker
from app.processors.chunkers.character import CharacterChunker
from app.processors.text_file import TextFileProcessor


def test_chunker_requires_implementation():
    class IncompleteChunker(BaseChunker):
        pass

    for cls in (BaseChunker, IncompleteChunker):
        with pytest.raises(TypeError, match='abstract'):
            cls()


@pytest.mark.parametrize('text,size,stride,expected', [
    ('', 4, 3, []),
    ('abc', 4, 3, [(0, 3, 'abc')]),
    ('abcd', 4, 3, [(0, 4, 'abcd')]),
    ('abcdefgh', 4, 4, [(0, 4, 'abcd'), (4, 8, 'efgh')]),
    ('abcdefgh', 4, 3, [(0, 4, 'abcd'), (3, 7, 'defg'), (6, 8, 'gh')]),
    ('abcdefg', 4, 3, [(0, 4, 'abcd'), (3, 7, 'defg')]),
    ('é🙂界x', 2, 1, [(0, 2, 'é🙂'), (1, 3, '🙂界'), (2, 4, '界x')]),
])
def test_character_windows(text, size, stride, expected):
    assert CharacterChunker().chunk(text, chunk_size=size, stride=stride) == [
        {'index': index, 'start': start, 'end': end, 'text': content}
        for index, (start, end, content) in enumerate(expected)
    ]


@pytest.mark.parametrize('size,stride', [(0, 1), (1, 0), (-1, 1), (1, 2),
                                       (True, 1), (1, False), (1.5, 1), (1, '1')])
def test_invalid_options_fail_before_reading(size, stride):
    from unittest.mock import Mock
    reader = Mock()
    with pytest.raises(ValidationError, match='positive integer'):
        TextFileProcessor(size, stride, reader=reader)
    reader.read.assert_not_called()
    with pytest.raises(ValidationError, match='positive integer'):
        CharacterChunker().chunk('', chunk_size=size, stride=stride)
