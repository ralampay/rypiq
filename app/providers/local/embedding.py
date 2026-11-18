import math
from pathlib import Path
from threading import Lock

from app.exceptions import (
    ProviderResponseError,
    ProviderInitializationError,
    ProviderRequestError,
    ProviderCleanupError,
)
from app.providers.contracts.embedding import EmbeddingProvider

class LocalEmbeddingProvider(EmbeddingProvider):
    """GGUF embedding provider using llama-cpp-python."""

    def __init__(
        self,
        model_path: str,
        context_size: int = 2048,
        gpu_layers: int = 0,
    ):
        path = Path(model_path).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                f"Embedding model not found: {path}"
            )

        self._model_name = path.name
        self._lock = Lock()
        try:
            # Import only when the local provider is constructed.
            from llama_cpp import Llama
            self._model = Llama(model_path=str(path), embedding=True,
                                n_ctx=context_size, n_gpu_layers=gpu_layers, verbose=False)
        except Exception:
            raise ProviderInitializationError('Local embedding model could not be loaded') from None

    @property
    def model_name(self) -> str:
        return self._model_name

    def embed(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        try:
            with self._lock:
                response = self._model.create_embedding(texts)
        except Exception:
            raise ProviderRequestError('Local embedding request failed') from None
        try:
            return self._vectors(response, len(texts))
        except ProviderResponseError:
            raise
        except (TypeError, ValueError, KeyError, IndexError, AttributeError):
            raise ProviderResponseError('Invalid embedding response from provider') from None

    @staticmethod
    def _vectors(response, count: int) -> list[list[float]]:
        data = sorted(
            response["data"],
            key=lambda item: item["index"]
        )

        if len(data) != count:
            raise ProviderResponseError(
                "Embedding count does not match the input count"
            )

        vectors = []
        dimensions = None

        for expected_index, item in enumerate(data):
            if item["index"] != expected_index:
                raise ProviderResponseError(
                    "Missing or duplicate embedding index"
                )

            vector = item["embedding"]

            if not isinstance(vector, list) or not vector:
                raise ProviderResponseError(
                    "Expected a nonempty embedding vector"
                )

            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in vector
            ):
                raise ProviderResponseError(
                    "Embedding contains invalid values"
                )

            if dimensions is None:
                dimensions = len(vector)
            elif len(vector) != dimensions:
                raise ProviderResponseError(
                    "Inconsistent embedding dimensions"
                )

            vectors.append(vector)

        return vectors

    def close(self) -> None:
        try:
            with self._lock:
                self._model.close()
        except Exception:
            raise ProviderCleanupError('Local embedding cleanup failed') from None
