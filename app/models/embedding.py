"""Shared validation for pgvector-compatible embeddings."""
import math
import struct
from numbers import Real


def validate_embedding(embedding, dimensions: int) -> list[float]:
    if type(dimensions) is not int or not 1 <= dimensions <= 16000:
        raise ValueError("Embedding size must be an integer between 1 and 16000")
    if not isinstance(embedding, (list, tuple)) or len(embedding) != dimensions:
        raise ValueError(f'Embedding must contain exactly {dimensions} numbers')
    try:
        if any(isinstance(x, bool) or not isinstance(x, Real) for x in embedding):
            raise ValueError
        # pgvector stores float32: validate the values PostgreSQL will receive.
        vector = [struct.unpack('f', struct.pack('f', x))[0] for x in embedding]
        if any(not math.isfinite(x) for x in vector) or not any(vector):
            raise ValueError
    except (TypeError, ValueError, OverflowError, struct.error):
        raise ValueError('Embedding must contain finite float32 values and have nonzero magnitude') from None
    return vector


