from app.models.processing import embedding_settings
import pytest

from app.operations.documents.search import semantic_search
from app.models.application import Application

APPLICATION = Application(name="test", embedding_model="fake", embedding_size=3)


class FakeEmbeddingProvider:
    model_name = "fake"

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
        application=APPLICATION,
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
        "application_id": APPLICATION.id,
        "metadata_filters": None,
        "expected_settings": embedding_settings(APPLICATION),
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
            application=APPLICATION,
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
            application=APPLICATION,
        )


def test_search_passes_metadata_filter_before_ranking():
    provider, repository = FakeEmbeddingProvider(), FakeRepository()
    semantic_search('query', provider, repository, application=APPLICATION,
                    metadata_filters={'topic': {'name': 'science'}})
    assert repository.received['metadata_filters'] == {'topic': {'name': 'science'}}


@pytest.mark.parametrize('filters', [[], 'tag', {'bad': float('nan')}])
def test_search_rejects_invalid_metadata_before_embedding(filters):
    provider, repository = FakeEmbeddingProvider(), FakeRepository()
    with pytest.raises(ValueError, match='JSON object'):
        semantic_search('query', provider, repository, application=APPLICATION, metadata_filters=filters)
    assert provider.inputs is None
    assert repository.received is None


@pytest.mark.parametrize('vectors', [None, [None], [[0, 0, 0]]])
def test_invalid_query_provider_output_is_typed(vectors):
    from unittest.mock import Mock
    from app.exceptions import ProviderResponseError
    from app.models.application import Application
    from app.operations.documents.search import SearchDocuments
    repository = Mock()
    provider = Mock(model_name='model', embed=Mock(return_value=vectors))
    with pytest.raises(ProviderResponseError):
        SearchDocuments(query='text', embedding_provider=provider, repository=repository,
                        application=Application(name='test', embedding_model='model', embedding_size=3)).execute()
    repository.search_by_vector.assert_not_called()
