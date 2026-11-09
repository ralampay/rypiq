"""Embed chunks extracted using text-file processing"""

from collections.abc import Callable

from app.embeddings.local import LocalEmbeddingProvider
from app.operations.text_file import process_text_file

DEFAULT_MODEL = "models/embeddinggemma-300m-Q4_0.gguf"
DEFAULT_PREFIX = "title: none | text: "

def embed_text_file(
    path: str,
    chunk_size: int,
    stride: int,
    model_path: str = DEFAULT_MODEL,
    context_size: int = 2048,
    gpu_layers: int = 0,
    batch_size: int = 1,
    provider_factory: Callable | None = None,
) -> dict:
    """Extract, chunk and embed a UTF-8 text file."""

    if (
        not isinstance(batch_size, int)
        or isinstance(batch_size, bool)
        or batch_size < 1
    ):
        raise ValueError(
            "batch_size must be a positive integer"
        )

    if context_size < 1:
        raise ValueError(
            "context_size must be positive"
        )

    # Reuse the existing text-file operation.
    document = process_text_file(
        path=path,
        chunk_size=chunk_size,
        stride=stride,
    )

    # Skip chunks containing only whitespace.
    chunks = [
        chunk
        for chunk in document["chunks"]
        if chunk["text"].strip()
    ]

    if not chunks:
        raise ValueError(
            "File has no nonempty text chunks"
        )

    factory = (
        provider_factory
        if provider_factory is not None
        else LocalEmbeddingProvider
    )

    # Load the GGUF model once for this operation
    provider = factory(
        model_path=model_path,
        context_size=context_size,
        gpu_layers=gpu_layers,
    )

    records = []
    dimensions = None

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]

        # EmbeddingGemma document formatting
        # TODO: abstract this later on for other gguf embedding models
        inputs = [
            DEFAULT_PREFIX + chunk["text"]
            for chunk in batch
        ]

        vectors = provider.embed(inputs)

        if len(vectors) != len(batch):
            raise ValueError("Embedding count mismatch")

        for chunk, vector in zip(
            batch, vectors, strict=True
        ):
            if not vector:
                raise ValueError("Empty embedding vector")

            if dimensions is None:
                dimensions = len(vector)

            elif len(vector) != dimensions:
                raise ValueError(
                    "Inconsistent vector dimensions"
                )

            records.append({
                **chunk,
                "embedding": vector,
            })

    return {
        **document,
        "model": provider.model_name,
        "dimensions": dimensions,
        "embedded_chunk_count": len(records),
        "chunks": records,
    }
