import math
from typing import Protocol


class EmbeddingProvider(Protocol):
    model_name: str
    dimensions: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def validate_vectors(vectors: list[list[float]], count: int, dimensions: int) -> None:
    if len(vectors) != count:
        raise ValueError("Embedding count mismatch")
    if type(dimensions) is not int or dimensions < 1:
        raise ValueError("Invalid embedding dimensions")
    for vector in vectors:
        if len(vector) != dimensions:
            raise ValueError("Embedding dimension mismatch")
        if any(type(value) not in (int, float) or not math.isfinite(value)
               for value in vector):
            raise ValueError("Embedding values must be finite numbers")
        if not any(vector):
            raise ValueError("Zero vectors cannot be used for cosine search")
