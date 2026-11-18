"""Factory for constructing embedding providers."""

from app.exceptions import ProviderConfigurationError
from app.providers.contracts.embedding import EmbeddingProvider
from app.providers.local.embedding import LocalEmbeddingProvider
from app.embeddings.config import DEFAULT_MODEL


def get_embedding_provider(
    provider: str = "local",
    model_path: str = DEFAULT_MODEL,
    context_size: int = 2048,
    gpu_layers: int = 0,
) -> EmbeddingProvider:

    if provider != "local":
        raise ProviderConfigurationError(
            f"Unsupported embedding provider: {provider}"
        )

    return LocalEmbeddingProvider(
        model_path=model_path,
        context_size=context_size,
        gpu_layers=gpu_layers,
    )
