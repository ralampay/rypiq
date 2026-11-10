from ragworkshop.embeddings.base import EmbeddingProvider, validate_vectors
from ragworkshop.processors.chunks import TextChunk


def embed_chunks(chunks: list[TextChunk], provider: EmbeddingProvider,
                 batch_size: int = 16) -> list[list[float]]:
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    vectors = []
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        output = provider.embed([chunk.text for chunk in batch])
        validate_vectors(output, len(batch), provider.dimensions)
        vectors.extend(output)
    return vectors
