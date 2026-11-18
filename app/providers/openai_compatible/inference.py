"""Chat adapter for OpenAI-compatible endpoints."""
from app.exceptions import ProviderRequestError, ValidationError
from app.providers.config import ProviderSettings
from app.providers.openai_compatible._client import ClientOwner
from app.providers.types.chat import ChatMessage, ChatResult
from app.providers.normalization import chat_response
from app.providers.validation import validate_chat


class OpenAICompatibleInferenceProvider(ClientOwner[ProviderSettings]):
    @property
    def model_name(self) -> str:
        return self.settings.model

    def chat(self, messages: list[ChatMessage], *, max_tokens: int = 512,
             temperature: float | None = None) -> ChatResult:
        serialized = validate_chat(messages, max_tokens, temperature)
        if self.settings.capability != 'inference':
            raise ValidationError('Provider was not configured for inference')
        options = {'max_completion_tokens': max_tokens}
        if temperature is not None:
            options['temperature'] = temperature
        try:
            response = self.client.chat.completions.create(model=self.model_name, messages=serialized,
                                                          **self._chat_options(options))
        except Exception:
            raise ProviderRequestError('Chat request failed; check provider availability and configuration') from None
        return chat_response(response, self.model_name)

    def _chat_options(self, options: dict) -> dict:
        return options
