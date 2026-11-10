from ragworkshop.embeddings.base import EmbeddingProvider
from ragworkshop.embeddings.fake import FakeEmbeddingProvider


def get_embedding_provider(name: str = "fake", model_path: str = "",
                           dimensions: int = 8) -> EmbeddingProvider:
    if name == "fake":
        if dimensions != 8:
            raise ValueError("Fake provider requires 8 dimensions")
        return FakeEmbeddingProvider()
    if name == "local":
        if not model_path:
            raise ValueError("Local provider requires an embedding model path")
        from ragworkshop.embeddings.local import LocalEmbeddingProvider
        return LocalEmbeddingProvider(model_path, dimensions)
    raise ValueError(f"Unsupported embedding provider: {name}")
