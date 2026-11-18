"""Chunking options shared by processors and chunkers."""
from app.exceptions import ValidationError


def validate_chunking(chunk_size: int, stride: int) -> None:
    if (not isinstance(chunk_size, int) or isinstance(chunk_size, bool)
            or not isinstance(stride, int) or isinstance(stride, bool)
            or chunk_size <= 0 or stride <= 0 or stride > chunk_size):
        raise ValidationError('Require positive integer parameters with stride <= chunk_size')
