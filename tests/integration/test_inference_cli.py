import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import cli
from app.database.backends.chroma import ChromaBackend
from app.embeddings.local import LocalEmbeddingProvider
from app.models.document import Document
from app.operations.embedding import DEFAULT_MODEL, DEFAULT_PREFIX


@pytest.mark.integration
def test_infer_cli_wiring(tmp_path, monkeypatch, capsys):
    model = tmp_path / "chat.gguf"
    model.touch()
    repository, provider = object(), object()
    monkeypatch.setattr(cli, "create_context", lambda: SimpleNamespace(documents=repository))
    factory = Mock(return_value=provider)
    operation = Mock(return_value={"response": "Answer [1]", "documents": [{"content": "Evidence"}]})
    monkeypatch.setattr(cli, "get_embedding_provider", factory)
    monkeypatch.setattr(cli, "infer", operation)
    monkeypatch.setattr(sys, "argv", ["rypiq", "infer", "--model", str(model),
                                     "--top-k", "3", "--query", "Question?",
                                     "--embedding-model", "embedding.gguf"])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert result["response"] == "Answer [1]"
    assert result["documents"][0]["content"] == "Evidence"
    factory.assert_called_once_with(model_path="embedding.gguf", gpu_layers=0)
    operation.assert_called_once_with(query="Question?", model_path=str(model),
                                      embedding_provider=provider, repository=repository,
                                      top_k=3, context_size=4096, gpu_layers=0, max_tokens=512)


@pytest.mark.integration
@pytest.mark.parametrize("options", [["--top-k", "0"], ["--query", " "],
                                     ["--max-tokens", "4096"]])
def test_invalid_cli_input(tmp_path, monkeypatch, capsys, options):
    model = tmp_path / "chat.gguf"
    model.touch()
    context = Mock()
    monkeypatch.setattr(cli, "create_context", context)
    monkeypatch.setattr(sys, "argv", ["rypiq", "infer", "--model", str(model),
                                     "--query", "Question?", *options])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "error:" in capsys.readouterr().err
    context.assert_not_called()


@pytest.mark.integration
def test_infer_cli_real_local_models(tmp_path, monkeypatch, capsys):
    model = Path(os.getenv("INFERENCE_MODEL_PATH", "models/Qwen3.5-0.8B-UD-Q6_K_XL.gguf"))
    embedding = Path(os.getenv("EMBEDDING_MODEL_PATH", DEFAULT_MODEL))
    if not model.is_file() or not embedding.is_file():
        pytest.skip("Local generation and embedding GGUF models are required")

    database = ChromaBackend({"path": str(tmp_path / "chroma")})
    database.create_documents_collection()
    repository = database.documents()
    provider = LocalEmbeddingProvider(str(embedding))
    contents = [
        "The Rypiq workshop access code is ORCHID-742.",
        "The Rypiq workshop starts at 9 AM in room Cedar.",
        "Participants should bring a laptop to the Rypiq workshop.",
        "Banana bread is baked with ripe bananas and flour.",
    ]
    vectors = provider.embed([DEFAULT_PREFIX + content for content in contents])
    documents = [repository.create(Document(content=content, embedding=vector,
                                            storage_key=f"documents/workshop-{i}.txt"))
                 for i, (content, vector) in enumerate(zip(contents, vectors))]
    monkeypatch.setattr(cli, "create_context", lambda: SimpleNamespace(documents=repository))
    monkeypatch.setattr(cli, "get_embedding_provider", lambda **kwargs: provider)
    monkeypatch.setattr(sys, "argv", ["rypiq", "infer", "--model", str(model),
                                     "--top-k", "3", "--query",
                                     "What is the Rypiq workshop access code?",
                                     "--max-tokens", "512"])
    cli.main()
    result = json.loads(capsys.readouterr().out)
    assert len(result["documents"]) == 3
    assert result["documents"][0]["id"] == str(documents[0].id)
    assert "ORCHID-742" in result["response"]
    assert all(source["content"] in contents for source in result["documents"])
    assert len(repository.list()) == 4
    # Keep a reviewable response when running pytest with --basetemp.
    (tmp_path / "inference-result.json").write_text(json.dumps(result, indent=2))
