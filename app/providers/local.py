"""Local inference through llama-cpp-python, independent of embedding resources."""
from pathlib import Path
from threading import Lock

from app.providers.config import ProviderSettings
from app.providers.contracts import ChatMessage, ChatResult, ProviderError
from app.providers.normalization import chat_response, validate_chat


class LocalInferenceProvider:
    def __init__(self, settings: ProviderSettings, *, model_factory=None) -> None:
        path = Path(settings.model).expanduser().resolve()
        if not path.is_file():
            raise ValueError('Local inference model file does not exist')
        if model_factory is None:
            from llama_cpp import Llama
            model_factory = Llama
        self.model_name = path.name
        self._lock = Lock()
        options = {} if settings.chat_format is None else {'chat_format': settings.chat_format}
        try:
            self._model = model_factory(model_path=str(path), embedding=False, n_ctx=settings.context_size,
                                        n_gpu_layers=settings.gpu_layers, verbose=False, **options)
        except Exception:
            raise ProviderError('Local inference model could not be loaded') from None

    def chat(self, messages: list[ChatMessage], *, max_tokens: int = 512,
             temperature: float | None = None) -> ChatResult:
        serialized = validate_chat(messages, max_tokens, temperature)
        options = {} if temperature is None else {'temperature': temperature}
        try:
            with self._lock:
                response = self._model.create_chat_completion(messages=serialized, max_tokens=max_tokens,
                                                               **options)
        except Exception:
            raise ProviderError('Local inference request failed') from None
        return chat_response(response, self.model_name)

    def close(self) -> None:
        try:
            with self._lock:
                self._model.close()
        except Exception:
            raise ProviderError('Local inference cleanup failed') from None
