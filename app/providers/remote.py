"""Compatibility facades for the original combined remote adapter API."""
from app.providers.config import ProviderSettings, DiscoverySettings
from app.providers.types.chat import ChatMessage, ChatResult
from app.providers.types.model import ModelInfo
from app.providers.openai_compatible._client import ClientOwner
from app.providers.openai import OpenAIEmbeddingProvider, OpenAIInferenceProvider, OpenAIModelDiscovery
from app.providers.llama_server import LlamaServerInferenceProvider


class OpenAIProvider(ClientOwner[ProviderSettings]):
    def __init__(self, settings: ProviderSettings, *, client, owns_client: bool = False) -> None:
        super().__init__(settings, client=client, owns_client=owns_client)

    @property
    def model_name(self) -> str:
        return self.settings.model

    def embed(self, texts: list[str]) -> list[list[float]]:
        return OpenAIEmbeddingProvider(self.settings, client=self.client).embed(texts)

    def chat(self, messages: list[ChatMessage], *, max_tokens: int = 512,
             temperature: float | None = None) -> ChatResult:
        adapter = OpenAIInferenceProvider(self.settings, client=self.client)
        adapter._chat_options = self._chat_options
        return adapter.chat(messages, max_tokens=max_tokens, temperature=temperature)

    def _chat_options(self, options: dict) -> dict:
        return options

    def list_models(self) -> list[ModelInfo]:
        settings = DiscoverySettings(self.settings.provider, self.settings.base_url,
                                     self.settings.api_key_env, self.settings.timeout)
        return OpenAIModelDiscovery(settings, client=self.client).list_models()


class LlamaServerProvider(OpenAIProvider):
    """Compatibility facade retaining llama.cpp token-limit mapping."""
    _chat_options = LlamaServerInferenceProvider._chat_options
