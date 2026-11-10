import os

import pytest

from ragworkshop.embeddings.factory import get_embedding_provider


@pytest.mark.real_model
def test_real_gguf():
    # Dedicated test variables survive the application-settings isolation fixture.
    path = os.getenv("RAG_WORKSHOP_TEST_MODEL")
    if not path:
        pytest.fail("Set RAG_WORKSHOP_TEST_MODEL when opting into real model tests")
    dimensions = int(os.environ["RAG_WORKSHOP_TEST_DIMENSIONS"])
    provider = get_embedding_provider("local", path, dimensions)
    try:
        first = provider.embed(["Python modules", "Vector retrieval"])
        assert len(first) == 2
        assert len(first[0]) == dimensions
        assert provider.embed(["Python modules"])[0] == pytest.approx(first[0], abs=1e-5)
    finally:
        provider.close()
