import sys
from types import SimpleNamespace

import pytest

from ragworkshop.embeddings.factory import get_embedding_provider


def test_local_without_model_runtime(monkeypatch, tmp_path):
    path = tmp_path / "placeholder.gguf"
    path.write_bytes(b"test fixture, not a model")
    class Model:
        def __init__(self, **kwargs):
            assert kwargs["embedding"] is True
        def create_embedding(self, texts):
            return {"data": [{"index": i, "embedding": [float(i + 1), 1.0]}
                             for i in reversed(range(len(texts)))]}
        def close(self):
            pass
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=Model))
    provider = get_embedding_provider("local", str(path), 2)
    assert provider.embed(["a", "b"]) == [[1.0, 1.0], [2.0, 1.0]]
    assert provider.embed([]) == []
    provider.close()
