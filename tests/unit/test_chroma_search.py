from uuid import uuid4

import pytest

from app.database.repositories.chroma import (
    ChromaDocumentRepository,
)
from app.models.document import Document


class ArrayLike(list):
    def tolist(self):
        return list(self)


class FakeCollection:

    def __init__(self):
        self.kwargs = None

    def query(self, **kwargs):
        self.kwargs = kwargs

        document = Document(
            id=uuid4(),
            content="Retrieved document",
            embedding=[0.1, 0.2],
            storage_key="documents/example.txt",
        )

        repository = ChromaDocumentRepository(self)

        return {
            "ids": [[str(document.id)]],
            "documents": [[document.content]],
            "embeddings": [[ArrayLike(document.embedding)]],
            "metadatas": [[
                repository._metadata(document)
            ]],
            "distances": [[0.12]],
        }


@pytest.mark.unit
def test_search_returns_documents_and_distances():
    collection = FakeCollection()

    repository = ChromaDocumentRepository(collection)

    results = repository.search_by_vector(
        embedding=[0.3, 0.4],
        limit=4,
    )

    assert len(results) == 1

    assert results[0].document.content == (
        "Retrieved document"
    )

    assert results[0].document.storage_key == (
        "documents/example.txt"
    )

    assert results[0].distance == pytest.approx(0.12)

    assert collection.kwargs["query_embeddings"] == [
        [0.3, 0.4]
    ]

    assert collection.kwargs["n_results"] == 4

    assert "where" not in collection.kwargs


@pytest.mark.unit
def test_search_supports_filters():
    collection = FakeCollection()

    repository = ChromaDocumentRepository(collection)

    repository.search_by_vector(
        embedding=[0.3, 0.4],
        filters={"embedding_model": "test-model"},
    )

    assert collection.kwargs["where"] == {
        "embedding_model": "test-model"
    }
