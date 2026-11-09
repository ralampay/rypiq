import pytest

from app.operations.search import semantic_search


class FakeEmbeddingProvider:

    def __init__(self):
        self.inputs = None

    def embed(self, texts):
        self.inputs = texts
        return [[0.1, 0.2, 0.3]]


class FakeRepository:

    def __init__(self):
        self.received = None

    def search_by_vector(self, **kwargs):
        self.received = kwargs
        return ["match"]


@pytest.mark.unit
def test_search_embeds_query_and_returns_results():
    provider = FakeEmbeddingProvider()
    repository = FakeRepository()

    results = semantic_search(
        query="What is RAG?",
        embedding_provider=provider,
        repository=repository,
        limit=4,
    )

    assert results == ["match"]

    assert provider.inputs == [
        "task: search result | query: What is RAG?"
    ]

    assert repository.received == {
        "embedding": [0.1, 0.2, 0.3],
        "limit": 4,
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    "query,limit",
    [
        ("", 4),
        ("   ", 4),
        ("valid", 0),
    ],
)
def test_search_rejects_invalid_inputs(query, limit):
    with pytest.raises(ValueError):
        semantic_search(
            query,
            FakeEmbeddingProvider(),
            FakeRepository(),
            limit,
        )


@pytest.mark.unit
def test_search_rejects_invalid_embedding():
    provider = FakeEmbeddingProvider()
    provider.embed = lambda texts: []

    with pytest.raises(ValueError):
        semantic_search(
            "test",
            provider,
            FakeRepository(),
        )
