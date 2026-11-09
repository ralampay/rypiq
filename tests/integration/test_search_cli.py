import json
import sys
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app import cli
from app.models.document import Document
from app.models.search import SearchResult


@pytest.mark.integration
def test_search_cli_uses_provider_and_repository(
    monkeypatch,
    capsys,
):
    repository = object()

    monkeypatch.setattr(
        cli,
        "create_context",
        lambda: SimpleNamespace(documents=repository),
    )

    received = {}

    def fake_factory(**kwargs):
        received["factory"] = kwargs
        return "fake-provider"

    def fake_search(**kwargs):
        received["search"] = kwargs

        return [
            SearchResult(
                document=Document(
                    id=uuid4(),
                    content="RAG retrieved chunk",
                    embedding=[0.1],
                    storage_key="documents/example.txt",
                ),
                distance=0.23,
            )
        ]

    monkeypatch.setattr(
        cli,
        "get_embedding_provider",
        fake_factory,
    )

    monkeypatch.setattr(
        cli,
        "semantic_search",
        fake_search,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rypiq",
            "search",
            "What is RAG?",
            "--limit",
            "4",
            "--model",
            "test.gguf",
        ],
    )

    cli.main()

    output = json.loads(capsys.readouterr().out)

    assert output[0]["content"] == "RAG retrieved chunk"
    assert output[0]["distance"] == 0.23

    assert received["factory"]["model_path"] == "test.gguf"

    assert received["search"] == {
        "query": "What is RAG?",
        "embedding_provider": "fake-provider",
        "repository": repository,
        "limit": 4,
    }


@pytest.mark.integration
def test_ingest_cli_uses_existing_operation(
    monkeypatch,
    capsys,
):
    context = SimpleNamespace(
        storage=object(),
        documents=object(),
    )

    monkeypatch.setattr(
        cli,
        "create_context",
        lambda: context,
    )

    received = {}

    def fake_ingest(**kwargs):
        received.update(kwargs)

        return [
            Document(
                content="Test",
                embedding=[0.1],
                storage_key="documents/example.txt",
            )
        ]

    monkeypatch.setattr(
        cli,
        "ingest_document",
        fake_ingest,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rypiq",
            "ingest",
            "samples/sample.txt",
            "--chunk_size",
            "100",
            "--stride",
            "50",
        ],
    )

    cli.main()

    output = json.loads(capsys.readouterr().out)

    assert output["indexed_chunks"] == 1
    assert received["storage"] is context.storage
    assert received["repository"] is context.documents
    assert received["chunk_size"] == 100
