import hashlib
from pathlib import Path

from ragworkshop.embeddings.base import validate_vectors


class LocalEmbeddingProvider:
    def __init__(self, model_path: str, dimensions: int):
        path = Path(model_path).expanduser().resolve(strict=True)
        if not path.is_file():
            raise ValueError("Model must be a regular file")
        if type(dimensions) is not int or dimensions < 1:
            raise ValueError("Invalid embedding dimensions")
        from llama_cpp import Llama
        with path.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        self.model_name = f"gguf:{fingerprint}:raw-v1"
        self.dimensions = dimensions
        self._model = Llama(model_path=str(path), embedding=True, n_gpu_layers=0,
                            n_ctx=2048, verbose=False)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._model.create_embedding(texts)
        data = sorted(response["data"], key=lambda item: item["index"])
        if [item["index"] for item in data] != list(range(len(texts))):
            raise ValueError("Missing or duplicate embedding index")
        vectors = [item["embedding"] for item in data]
        validate_vectors(vectors, len(texts), self.dimensions)
        return vectors

    def close(self) -> None:
        self._model.close()
