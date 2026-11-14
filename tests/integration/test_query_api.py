import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import sys

from fastapi.testclient import TestClient
import pytest

import app
from app import cli
from app.controllers import query_controller
from app.database.backends.chroma import ChromaBackend
from app.embeddings.local import LocalEmbeddingProvider
from app.models.document import Document
from app.operations.embedding import DEFAULT_MODEL, DEFAULT_PREFIX


@pytest.fixture
def api_setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = tmp_path / "model.gguf"
    model.touch()
    (tmp_path / ".env").write_text(f"INFERENCE_MODEL={model}\nEMBEDDING_MODEL={model}\n")
    monkeypatch.delenv("INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    repository = object()
    provider = Mock()
    monkeypatch.setattr(app, "create_context", lambda: SimpleNamespace(documents=repository))
    factory = Mock(return_value=provider)
    monkeypatch.setattr(app, "get_embedding_provider", factory)
    operation = Mock(return_value={
        "response": "Answer [1]",
        "documents": [{"citation": 1, "id": "source-id", "distance": 0.1,
                       "content": "Evidence", "storage_key": "documents/source.txt",
                       "filename": "source.txt", "chunk_index": 0}],
    })
    monkeypatch.setattr(query_controller, "infer", operation)
    return app.create_app(), operation, provider, repository, model, factory


@pytest.mark.integration
def test_query_cors_preflight(api_setup):
    application, operation, *_ = api_setup
    with TestClient(application) as client:
        response = client.options("/query?top_k=3", headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        })
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "*"
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "content-type" in response.headers["access-control-allow-headers"].lower()
    operation.assert_not_called()


@pytest.mark.integration
@pytest.mark.parametrize("status", [200, 422, 503])
def test_query_cors_response(api_setup, status):
    application, operation, *_ = api_setup
    if status == 503:
        operation.side_effect = RuntimeError("Inference unavailable")
    body = {} if status == 422 else {"query": "question"}
    with TestClient(application) as client:
        response = client.post("/query", json=body,
                               headers={"Origin": "http://localhost:5173"})
    assert response.status_code == status
    assert response.headers["access-control-allow-origin"] == "*"
    assert "access-control-allow-credentials" not in response.headers


@pytest.mark.integration
@pytest.mark.parametrize("suffix,expected_k", [("", 3), ("?top_k=5", 5)])
def test_query_uses_dotenv_and_passes_top_k(api_setup, suffix, expected_k):
    application, operation, provider, repository, model, factory = api_setup
    with TestClient(application) as client:
        response = client.post("/query" + suffix, json={"query": " Question? "})
        assert response.status_code == 200
        assert response.json() == operation.return_value
        operation.assert_called_once_with(query="Question?", model_path=str(model),
                                          embedding_provider=provider, repository=repository,
                                          top_k=expected_k)
        factory.assert_called_once_with(model_path=str(model))
        schema = client.get("/openapi.json").json()
        assert "security" not in schema["paths"]["/query"]["post"]
    provider.close.assert_called_once()


@pytest.mark.integration
@pytest.mark.parametrize("suffix,body", [
    ("?top_k=0", {"query": "question"}),
    ("?top_k=-1", {"query": "question"}),
    ("?top_k=1.5", {"query": "question"}),
    ("?top_k=abc", {"query": "question"}),
    ("", {}), ("", {"query": "  "}), ("", {"query": 3}),
    ("", {"query": "question", "top_k": 2}),
])
def test_query_validation(api_setup, suffix, body):
    application, operation, *_ = api_setup
    with TestClient(application) as client:
        assert client.post("/query" + suffix, json=body).status_code == 422
    operation.assert_not_called()


@pytest.mark.integration
@pytest.mark.parametrize("failure,status", [(ValueError("too long"), 400),
                                           (RuntimeError("private model path"), 503),
                                           (OSError("private path"), 503)])
def test_query_errors(api_setup, failure, status):
    application, operation, *_ = api_setup
    operation.side_effect = failure
    with TestClient(application) as client:
        response = client.post("/query", json={"query": "question"})
        assert response.status_code == status
        assert str(failure) not in response.text


@pytest.mark.integration
def test_empty_retrieval(api_setup):
    application, operation, *_ = api_setup
    operation.return_value = {"response": "No indexed documents matched the query.", "documents": []}
    with TestClient(application) as client:
        response = client.post("/query", json={"query": "question"})
        assert response.status_code == 200
        assert response.json()["documents"] == []


@pytest.mark.integration
@pytest.mark.parametrize("value", ["", "/missing/model.gguf"])
def test_bad_server_configuration_fails_at_startup(tmp_path, monkeypatch, value):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("INFERENCE_MODEL", value)
    with pytest.raises(ValueError, match="INFERENCE_MODEL"):
        with TestClient(app.create_app()):
            pass


@pytest.mark.integration
def test_exported_environment_takes_precedence(api_setup, tmp_path, monkeypatch):
    application, operation, *_ = api_setup
    override = tmp_path / "override.gguf"
    override.touch()
    monkeypatch.setenv("INFERENCE_MODEL", str(override))
    with TestClient(app.create_app()) as client:
        client.post("/query", json={"query": "question"})
    assert operation.call_args.kwargs["model_path"] == str(override)


@pytest.mark.integration
def test_server_cli(monkeypatch):
    import uvicorn

    run = Mock()
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setattr(sys, "argv", ["rypiq", "server", "--port", "8100", "--reload"])
    cli.main()
    run.assert_called_once_with("app:create_app", factory=True, host="127.0.0.1",
                                port=8100, reload=True)


@pytest.mark.integration
def test_query_with_real_local_models(tmp_path, monkeypatch):
    model = Path(os.getenv("INFERENCE_MODEL_PATH", "models/Qwen3.5-0.8B-UD-Q6_K_XL.gguf")).resolve()
    embedding = Path(os.getenv("EMBEDDING_MODEL_PATH", DEFAULT_MODEL)).resolve()
    if not model.is_file() or not embedding.is_file():
        pytest.skip("Local generation and embedding GGUF models are required")
    database = ChromaBackend({"path": str(tmp_path / "chroma")})
    database.create_documents_collection()
    repository = database.documents()
    provider = LocalEmbeddingProvider(str(embedding))
    content = "The Rypiq workshop access code is ORCHID-742."
    try:
        vector = provider.embed([DEFAULT_PREFIX + content])[0]
    finally:
        provider.close()
    document = repository.create(Document(content=content, embedding=vector,
                                          storage_key="documents/workshop.txt"))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("INFERENCE_MODEL", raising=False)
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    (tmp_path / ".env").write_text(f"INFERENCE_MODEL={model}\nEMBEDDING_MODEL={embedding}\n")
    monkeypatch.setattr(app, "create_context", lambda: SimpleNamespace(documents=repository))
    with TestClient(app.create_app()) as client:
        response = client.post("/query?top_k=1", json={"query": "What is the Rypiq workshop access code?"})
        assert response.status_code == 200, response.text
        result = response.json()
        assert "ORCHID-742" in result["response"]
        assert result["documents"][0]["id"] == str(document.id)
        assert len(result["documents"]) == 1
        (tmp_path / "api-result.json").write_text(json.dumps(result, indent=2))
