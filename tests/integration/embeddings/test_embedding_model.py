import math
import os
from pathlib import Path

import pytest

from app.embeddings.local import LocalEmbeddingProvider

MODEL_PATH = Path(
    os.getenv(
        "EMBEDDING_MODEL_PATH",
        "models/embeddinggemma-300m-Q4_0.gguf",
    )
)

@pytest.fixture(scope="module")
def provider():
    if not MODEL_PATH.is_file():
        pytest.skip("Embedding model not available")

    return LocalEmbeddingProvider(
        model_path=str(MODEL_PATH),
        context_size=2048,
        gpu_layers=0,
    )

@pytest.mark.integration
def test_model_embedding_dimensions(provider):
    vectors = provider.embed([
        "title: none | text: Machine learning users data."
    ])

    assert len(vectors) == 1
    assert len(vectors[0]) == 768 # for gemma embedding

    assert all(
        math.isfinite(value)
        for value in vectors[0]
    )

@pytest.mark.integration
def test_model_multiple_inputs(provider):
    texts = [
        "title: none | text: Cats are domestic animals.",
        "title: none | text: Dogs are domestic animals.",
    ]

    vectors = provider.embed(texts)

    assert len(vectors) == 2
    assert all(len(vector) == 768 for vector in vectors)

@pytest.mark.integration
def test_model_repeated_input(provider):
    text = "title; none | text: Machine learning is useful."

    first = provider.embed([text])[0]
    second = provider.embed([text])[0]

    assert first == pytest.approx(second, abs=1e-5)
