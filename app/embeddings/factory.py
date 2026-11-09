"""Factory for constructing embedding providers."""

from app.embeddings.base import EmbeddingProvider
from app.embeddings.local import LocalEmbeddingProvider
from app.operations.embedding import DEFAULT_MODEL


def get_embedding_provider(
    provider: str = "local",
    model_path: str = DEFAULT_MODEL,
    context_size: int = 2048,
    gpu_layers: int = 0,
) -> EmbeddingProvider:

    if provider != "local":
        raise ValueError(
            f"Unsupported embedding provider: {provider}"
        )

    return LocalEmbeddingProvider(
        model_path=model_path,
        context_size=context_size,
        gpu_layers=gpu_layers,
    )
