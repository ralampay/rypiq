"""Coordinate file reading and chunking while preserving the processing result."""
from pathlib import Path

from app.processors.base import BaseProcessor
from app.processors.chunkers.base import BaseChunker
from app.processors.chunkers.character import CharacterChunker
from app.processors.chunkers._validation import validate_chunking
from app.processors.file_reader import FileTextReader


class TextFileProcessor(BaseProcessor):
    def __init__(self, chunk_size: int, stride: int, *,
                 reader: FileTextReader | None = None,
                 chunker: BaseChunker | None = None) -> None:
        validate_chunking(chunk_size, stride)
        self.chunk_size = chunk_size
        self.stride = stride
        self.reader = reader if reader is not None else FileTextReader()
        self.chunker = chunker if chunker is not None else CharacterChunker()

    def process(self, content: str) -> dict:
        path = Path(content)
        text = self.reader.read(path)
        chunks = self.chunker.chunk(text, chunk_size=self.chunk_size, stride=self.stride)

        return {
            "medium": "text-file",
            "source": str(path),
            "chunk_size": self.chunk_size,
            "stride": self.stride,
            "total_characters": len(text),
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
