import pytest

from app.embeddings import factory


@pytest.mark.unit
def test_factory_constructs_local_provider(monkeypatch):
    received = {}

    class FakeEmbeddingProvider:
        def __init__(self, **kwargs):
            received.update(kwargs)

    monkeypatch.setattr(
        factory,
        "LocalEmbeddingProvider",
        FakeEmbeddingProvider,
    )

    provider = factory.get_embedding_provider(
        model_path="fake.gguf",
        context_size=128,
        gpu_layers=3,
    )

    assert isinstance(provider, FakeEmbeddingProvider)

    assert received == {
        "model_path": "fake.gguf",
        "context_size": 128,
        "gpu_layers": 3,
    }


@pytest.mark.unit
def test_factory_rejects_unknown_provider():
    with pytest.raises(
        ValueError,
        match="Unsupported",
    ):
        factory.get_embedding_provider(
            provider="online",
        )
