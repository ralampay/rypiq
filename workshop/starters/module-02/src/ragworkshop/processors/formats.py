import json
from collections.abc import Callable
from typing import Protocol
from ragworkshop.processors.chunks import Metadata, TextChunk, TextChunker

class TextProcessor(Protocol):

    def process(self, text: str, metadata: Metadata | None=None) -> list[TextChunk]:
        ...

class PlainTextProcessor:

    def __init__(self, chunker: TextChunker):
        self.chunker = chunker

    def process(self, text: str, metadata: Metadata | None=None) -> list[TextChunk]:
        raise NotImplementedError('TODO: PlainTextProcessor.process')

class MarkdownTextProcessor:
    """Preserve Markdown syntax; offsets refer to the Markdown source."""

    def __init__(self, chunker: TextChunker):
        self.chunker = chunker

    def process(self, text: str, metadata: Metadata | None=None) -> list[TextChunk]:
        raise NotImplementedError('TODO: MarkdownTextProcessor.process')

class JsonTextProcessor:

    def __init__(self, chunker: TextChunker, field: str='text'):
        self.chunker = chunker
        self.field = field

    def process(self, text: str, metadata: Metadata | None=None) -> list[TextChunk]:
        raise NotImplementedError('TODO: JsonTextProcessor.process')
Builder = Callable[[TextChunker, str], TextProcessor]
PROCESSORS: dict[str, Builder] = {'plain': lambda chunker, field: PlainTextProcessor(chunker), 'markdown': lambda chunker, field: MarkdownTextProcessor(chunker), 'json': lambda chunker, field: JsonTextProcessor(chunker, field)}

def get_processor(name: str, chunker: TextChunker, field: str='text') -> TextProcessor:
    raise NotImplementedError('TODO: get_processor')
