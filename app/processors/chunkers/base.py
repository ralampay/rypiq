"""Contract for splitting extracted text into indexed spans."""
from abc import ABC, abstractmethod


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, text: str, *, chunk_size: int, stride: int) -> list[dict]:
        """Return chunks with index, start, end, and text fields."""
        pass
