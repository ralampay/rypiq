"""Text-only inference using borrowed provider resources."""
from app.providers.types.chat import ChatMessage, ChatResult
from app.providers.contracts.inference import InferenceProvider
from app.providers.validation import validate_chat


class Chat:
    def __init__(self, *, messages: list[ChatMessage], provider: InferenceProvider,
                 max_tokens: int = 512, temperature: float | None = None) -> None:
        self.messages = messages
        self.provider = provider
        self.max_tokens = max_tokens
        self.temperature = temperature

    def execute(self) -> ChatResult:
        validate_chat(self.messages, self.max_tokens, self.temperature)
        return self.provider.chat(self.messages, max_tokens=self.max_tokens, temperature=self.temperature)
