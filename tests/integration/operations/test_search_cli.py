import json
import sys
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app import cli, application
from app.models.document import Document
from app.models.search import SearchResult


@pytest.mark.integration
def test_search_cli_uses_provider_and_repository(
    monkeypatch,
    capsys,
):
    received = {}

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

    monkeypatch.setattr(application, 'search_documents', fake_search)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rypiq",
            "search",
            "What is RAG?",
            "--application-id",
            "00000000-0000-0000-0000-000000000001",
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

    assert received['search'] == {
        'application_id': UUID(int=1),
        'query': 'What is RAG?', 'limit': 4, 'model_path': 'test.gguf',
        'context_size': None, 'gpu_layers': None,
    }


@pytest.mark.integration
def test_ingest_cli_uses_existing_operation(
    monkeypatch,
    capsys,
):
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
        application,
        "ingest_file",
        fake_ingest,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rypiq",
            "ingest",
            "samples/sample.txt",
            "--application-id",
            "00000000-0000-0000-0000-000000000001",
            "--chunk_size",
            "100",
            "--stride",
            "50",
        ],
    )

    cli.main()

    output = json.loads(capsys.readouterr().out)

    assert output["indexed_chunks"] == 1
    assert received["chunk_size"] == 100
