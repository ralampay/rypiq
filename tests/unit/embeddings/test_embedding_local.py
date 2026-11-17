import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.embeddings.base import EmbeddingProvider
from app.embeddings.local import LocalEmbeddingProvider

@pytest.fixture
def mock_llama(monkeypatch):
    """Replace llama_cpp with a fake inference module."""

    instance = MagicMock()

    instance.create_embedding.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2, 0.3]},
            {"index": 1, "embedding": [0.4, 0.5, 0.6]},
        ]
    }

    constructor = MagicMock(return_value=instance)

    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(Llama=constructor),
    )

    return constructor, instance

@pytest.fixture
def model_path(tmp_path):
    """Create a placeholder GGUF file."""
    path = tmp_path / "test-model.gguf"
    path.touch()

    return path

@pytest.mark.unit
def test_local_provider_implements_interface(
    mock_llama, model_path
):
    provider = LocalEmbeddingProvider(str(model_path))

    assert isinstance(provider, EmbeddingProvider)

@pytest.mark.unit
def test_model_is_loaded_for_embeddings(
    mock_llama, model_path
):
    constructor, _ = mock_llama

    LocalEmbeddingProvider(
        str(model_path),
        context_size=2048,
        gpu_layers=0,
    )

    constructor.assert_called_once_with(
        model_path=str(model_path),
        embedding=True,
        n_ctx=2048,
        n_gpu_layers=0,
        verbose=False,
    )

@pytest.mark.unit
def test_model_name(mock_llama, model_path):
    provider = LocalEmbeddingProvider(str(model_path))

    assert provider.model_name == "test-model.gguf"

@pytest.mark.unit
def test_missing_model_raises_error(mock_llama, tmp_path):
    missing = tmp_path / "missing.gguf"

    with pytest.raises(FileNotFoundError):
        LocalEmbeddingProvider(str(missing))


@pytest.mark.unit
def test_embedding_generation(mock_llama, model_path):
    _, instance = mock_llama

    provider = LocalEmbeddingProvider(str(model_path))

    texts = ["First sentence", "Second sentence"]
    result = provider.embed(texts)

    instance.create_embedding.assert_called_once_with(texts)

    assert result == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

@pytest.mark.unit
def test_empty_input(mock_llama, model_path):
    _, instance = mock_llama

    provider = LocalEmbeddingProvider(str(model_path))

    assert provider.embed([]) == []

    instance.create_embedding.assert_not_called()

@pytest.mark.unit
def test_output_order(mock_llama, model_path):
    _, instance  = mock_llama

    instance.create_embedding.return_value = {
        "data": [
            {"index": 1, "embedding": [0.4, 0.5, 0.6]},
            {"index": 0, "embedding": [0.1, 0.2, 0.3]},
        ]
    }

    provider = LocalEmbeddingProvider(str(model_path))

    vectors = provider.embed(["first", "second"])

    assert vectors == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

@pytest.mark.unit
def test_invalid_vector_count(mock_llama, model_path):
    _, instance = mock_llama

    instance.create_embedding.return_value =  {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2]}
        ]
    }

    provider = LocalEmbeddingProvider(str(model_path))

    with pytest.raises(ValueError):
        provider.embed(["first", "second"])

@pytest.mark.unit
def test_invalid_embedding_dimensions(mock_llama, model_path):
    _, instance = mock_llama

    instance.create_embedding.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2]},
            {"index": 1, "embedding": [0.3, 0.4, 0.5]},
        ]
    }

    provider = LocalEmbeddingProvider(str(model_path))

    with pytest.raises(ValueError):
        provider.embed(["first", "second"])

@pytest.mark.unit
def test_nonfinite_embedding_values(mock_llama, model_path):
    _, instance = mock_llama
    
    instance.create_embedding.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, float("nan")]},
        ]
    }

    provider = LocalEmbeddingProvider(str(model_path))

    with pytest.raises(ValueError):
        provider.embed(["first"])

@pytest.mark.unit
def test_model_is_reused(mock_llama, model_path):
    constructor, instance = mock_llama

    provider = LocalEmbeddingProvider(str(model_path))

    instance.create_embedding.return_value = {
        "data": [
            {"index": 0, "embedding": [0.1, 0.2, 0.3]}
        ]
    }

    provider.embed(["first"])
    provider.embed(["second"])

    constructor.assert_called_once()
    assert instance.create_embedding.call_count == 2
