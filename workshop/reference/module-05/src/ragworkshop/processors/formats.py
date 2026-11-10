import json
from collections.abc import Callable
from typing import Protocol

from ragworkshop.processors.chunks import Metadata, TextChunk, TextChunker


class TextProcessor(Protocol):
    def process(self, text: str, metadata: Metadata | None = None) -> list[TextChunk]: ...


class PlainTextProcessor:
    def __init__(self, chunker: TextChunker):
        self.chunker = chunker

    def process(self, text: str, metadata: Metadata | None = None) -> list[TextChunk]:
        return self.chunker.chunk(text, metadata)


class MarkdownTextProcessor:
    """Preserve Markdown syntax; offsets refer to the Markdown source."""
    def __init__(self, chunker: TextChunker):
        self.chunker = chunker

    def process(self, text: str, metadata: Metadata | None = None) -> list[TextChunk]:
        return self.chunker.chunk(text, {**(metadata or {}), "format": "markdown"})


class JsonTextProcessor:
    def __init__(self, chunker: TextChunker, field: str = "text"):
        self.chunker = chunker
        self.field = field

    def process(self, text: str, metadata: Metadata | None = None) -> list[TextChunk]:
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError("Invalid JSON text") from exc
        if not isinstance(value, dict) or not isinstance(value.get(self.field), str):
            raise ValueError("JSON field must contain a string")
        return self.chunker.chunk(value[self.field],
                                  {**(metadata or {}), "field": self.field})


Builder = Callable[[TextChunker, str], TextProcessor]
PROCESSORS: dict[str, Builder] = {
    "plain": lambda chunker, field: PlainTextProcessor(chunker),
    "markdown": lambda chunker, field: MarkdownTextProcessor(chunker),
    "json": lambda chunker, field: JsonTextProcessor(chunker, field),
}


def get_processor(name: str, chunker: TextChunker, field: str = "text") -> TextProcessor:
    try:
        builder = PROCESSORS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported text format: {name}") from exc
    return builder(chunker, field)
