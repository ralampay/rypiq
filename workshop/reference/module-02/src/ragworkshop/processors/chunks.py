from dataclasses import dataclass, field

Metadata = dict[str, str | int | float | bool]


@dataclass(frozen=True)
class TextChunk:
    index: int
    start: int
    end: int
    text: str
    metadata: Metadata = field(default_factory=dict)


class TextChunker:
    def __init__(self, chunk_size: int = 200, stride: int = 150):
        if (type(chunk_size) is not int or type(stride) is not int
                or not 0 < stride <= chunk_size):
            raise ValueError("Require integer 0 < stride <= chunk_size")
        self.chunk_size = chunk_size
        self.stride = stride

    def chunk(self, text: str, metadata: Metadata | None = None) -> list[TextChunk]:
        chunks = []
        for start in range(0, len(text), self.stride):
            end = min(start + self.chunk_size, len(text))
            chunks.append(TextChunk(len(chunks), start, end, text[start:end],
                                    dict(metadata or {})))
            if end == len(text):
                break
        return chunks
