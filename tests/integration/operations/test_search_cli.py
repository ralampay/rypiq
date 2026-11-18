import json
import sys
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest

from app import cli, application
from app.models.document import Document, DocumentDetails
from app.models.embedding import Embedding
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
                    storage_key="documents/example.txt",
                    name=str(uuid4()),
                ),
                embedding=Embedding(document_id=uuid4(), content="RAG retrieved chunk", embedding=[0.1]),
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
        'context_size': None, 'gpu_layers': None, 'metadata_filters': None,
    }


@pytest.mark.integration
def test_embeddings_create_cli_dispatch(
    monkeypatch,
    capsys,
):
    received = {}

    def fake_ingest(**kwargs):
        received.update(kwargs)

        parent = Document(storage_key='documents/example.txt', name=str(uuid4()))
        return DocumentDetails(parent, [Embedding(document_id=parent.id, content='Test', embedding=[0.1])])

    monkeypatch.setattr(
        application,
        "create_document",
        fake_ingest,
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rypiq",
            "documents:create", "--name", "test document",
            "samples/sample.txt",
            "--application-id",
            "00000000-0000-0000-0000-000000000001",
        ],
    )

    cli.main()

    output = json.loads(capsys.readouterr().out)

    assert output["embedding_count"] == 1
    assert received["batch_size"] == 1
    assert received["application_id"] == UUID(int=1)


def test_search_cli_passes_json_metadata_filter(monkeypatch, capsys):
    from unittest.mock import Mock
    search = Mock(return_value=[])
    monkeypatch.setattr(application, 'search_documents', search)
    cli.main(['search', 'query', '--application-id', str(UUID(int=1)),
              '--metadata-filter', '{"start":0}'])
    assert search.call_args.kwargs['metadata_filters'] == {'start': 0}
    assert json.loads(capsys.readouterr().out) == []
