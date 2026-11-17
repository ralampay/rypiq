from pathlib import Path

from app.processors.base import BaseProcessor
from app.processors.extraction import extract_file

class TextFileProcessor(BaseProcessor):
    def __init__(self, chunk_size: int, stride: int):
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or not isinstance(stride, int)
            or isinstance(stride, bool)
            or chunk_size <= 0
            or stride <= 0
            or stride > chunk_size
        ):
            raise ValueError(
                "Require positive integer parameters "
                "with stride <= chunk_size"
            )

        self.chunk_size = chunk_size
        self.stride = stride

    def process(self, content: str) -> dict:
        path = Path(content)

        text = extract_file(path)

        chunks = []

        for start in range(0, len(text), self.stride):
            end = min(start + self.chunk_size, len(text))

            chunks.append({
                "index": len(chunks),
                "start": start,
                "end": end,
                "text": text[start:end],
            })

            if end == len(text):
                break

        return {
            "medium": "text-file",
            "source": str(path),
            "chunk_size": self.chunk_size,
            "stride": self.stride,
            "total_characters": len(text),
            "chunk_count": len(chunks),
            "chunks": chunks,
        }
