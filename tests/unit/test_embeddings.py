import inspect
import pytest

from app.embeddings.base import EmbeddingProvider

class MockEmbeddingProvider(EmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "mock-model"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [float(len(text)), 0.0, 1.0] for text in texts

        ]

@pytest.fixture
def provider():
    return MockEmbeddingProvider()

@pytest.mark.unit
def test_embedding_provider_is_abstract():
    assert inspect.isabstract(EmbeddingProvider)

    with pytest.raises(TypeError):
        EmbeddingProvider()

@pytest.mark.unit
def test_incomplete_provider_cannot_be_instantiated():
    class IncompleteProvider(EmbeddingProvider):
        pass

    with pytest.raises(TypeError):
        IncompleteProvider()

@pytest.mark.unit
def test_provider_requires_model_name():
    class MissingModelNameProvider(EmbeddingProvider):
        def embed(self, texts: list[str]) -> list[list[float]]:
            return []

    with pytest.raises(TypeError):
        MissingModelNameProvider()

@pytest.mark.unit
def test_provider_requires_embed():
    class MissingEmbedProvider(EmbeddingProvider):
        @property
        def model_name(self) -> str:
            return "test-model"

    with pytest.raises(TypeError):
        MissingEmbedProvider()

@pytest.mark.unit
def test_concrete_provider_can_be_instantiated(provider):
    assert isinstance(provider, EmbeddingProvider)
    assert provider.model_name == "mock-model"

@pytest.mark.unit
def test_concrete_provider_generates_embeddings(provider):
    vectors = provider.embed(["hello", "world!"])

    assert vectors == [
        [5.0, 0.0, 1.0],
        [6.0, 0.0, 1.0],
    ]
