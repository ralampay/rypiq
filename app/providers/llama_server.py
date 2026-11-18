"""LlamaServer capability adapters."""
from app.providers.openai_compatible.embedding import OpenAICompatibleEmbeddingProvider
from app.providers.openai_compatible.inference import OpenAICompatibleInferenceProvider
from app.providers.openai_compatible.discovery import OpenAICompatibleModelDiscovery


class LlamaServerEmbeddingProvider(OpenAICompatibleEmbeddingProvider):
    """LlamaServer EmbeddingProvider adapter."""


class LlamaServerInferenceProvider(OpenAICompatibleInferenceProvider):
    """llama.cpp uses max_tokens for its token limit."""
    def _chat_options(self, options: dict) -> dict:
        return {'max_tokens' if key == 'max_completion_tokens' else key: value for key, value in options.items()}


class LlamaServerModelDiscovery(OpenAICompatibleModelDiscovery):
    """LlamaServer ModelDiscovery adapter."""
