"""Separate remote adapters sharing the OpenAI-compatible HTTP protocol."""
from app.providers.config import ProviderSettings
from app.providers.contracts import ChatMessage, ChatResult, ModelInfo, ProviderError
from app.providers.normalization import chat_response, embedding_response, validate_chat, validate_texts


class OpenAIProvider:
    def __init__(self, settings: ProviderSettings, *, client, owns_client: bool = False) -> None:
        self.settings = settings
        self.client = client
        self.owns_client = owns_client

    @property
    def model_name(self) -> str:
        return self.settings.model

    def embed(self, texts: list[str]) -> list[list[float]]:
        validate_texts(texts)
        if self.settings.capability != 'embedding':
            raise ValueError('Provider was not configured for embeddings')
        if not texts:
            return []
        options = {}
        if self.settings.dimensions is not None:
            options['dimensions'] = self.settings.dimensions
        try:
            response = self.client.embeddings.create(model=self.model_name, input=texts, **options)
        except Exception:
            raise ProviderError('Embedding request failed; check provider availability and configuration') from None
        return embedding_response(response, len(texts), self.settings.embedding_size)

    def chat(self, messages: list[ChatMessage], *, max_tokens: int = 512,
             temperature: float | None = None) -> ChatResult:
        serialized = validate_chat(messages, max_tokens, temperature)
        if self.settings.capability != 'inference':
            raise ValueError('Provider was not configured for inference')
        options = {'max_completion_tokens': max_tokens}
        if temperature is not None:
            options['temperature'] = temperature
        try:
            response = self.client.chat.completions.create(model=self.model_name, messages=serialized,
                                                          **self._chat_options(options))
        except Exception:
            raise ProviderError('Chat request failed; check provider availability and configuration') from None
        return chat_response(response, self.model_name)

    def _chat_options(self, options: dict) -> dict:
        return options

    def list_models(self) -> list[ModelInfo]:
        try:
            response = self.client.models.list()
            data = response['data'] if isinstance(response, dict) else [item.model_dump() for item in response]
            ids = {item['id'] for item in data}
            if any(not isinstance(model, str) or not model.strip() for model in ids):
                raise ValueError
            return [ModelInfo(model, self.settings.provider, (), 'discovered', 'available', model)
                    for model in sorted(ids)]
        except Exception:
            raise ProviderError('Model discovery failed; check provider availability and configuration') from None

    def close(self) -> None:
        if self.owns_client:
            try:
                self.client.close()
            except Exception:
                raise ProviderError('Provider client cleanup failed') from None


class LlamaServerProvider(OpenAIProvider):
    """llama.cpp uses max_tokens rather than OpenAI's newer token-limit field."""
    def _chat_options(self, options: dict) -> dict:
        return {'max_tokens' if key == 'max_completion_tokens' else key: value for key, value in options.items()}
