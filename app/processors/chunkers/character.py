"""Overlapping character windows with offsets into the extracted text."""
from .base import BaseChunker
from ._validation import validate_chunking


class CharacterChunker(BaseChunker):
    def chunk(self, text: str, *, chunk_size: int, stride: int) -> list[dict]:
        validate_chunking(chunk_size, stride)
        chunks = []

        for start in range(0, len(text), stride):
            end = min(start + chunk_size, len(text))

            chunks.append({
                "index": len(chunks),
                "start": start,
                "end": end,
                "text": text[start:end],
            })

            if end == len(text):
                break

        return chunks
