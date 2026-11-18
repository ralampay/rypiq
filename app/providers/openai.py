"""OpenAI capability adapters."""
from app.providers.openai_compatible.embedding import OpenAICompatibleEmbeddingProvider
from app.providers.openai_compatible.inference import OpenAICompatibleInferenceProvider
from app.providers.openai_compatible.discovery import OpenAICompatibleModelDiscovery


class OpenAIEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """OpenAI EmbeddingProvider adapter."""


class OpenAIInferenceProvider(OpenAICompatibleInferenceProvider):
    """OpenAI InferenceProvider adapter."""


class OpenAIModelDiscovery(OpenAICompatibleModelDiscovery):
    """OpenAI ModelDiscovery adapter."""
