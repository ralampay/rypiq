import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.models.document import Document
from app.models.search import SearchResult
from app.operations.inference import infer
from app.operations.search import QUERY_PREFIX


@pytest.fixture
def setup_inference(tmp_path, monkeypatch):
    path = tmp_path / "model.gguf"
    path.touch()
    provider = Mock()
    provider.embed.return_value = [[0.1, 0.2]]
    repository = Mock()
    repository.search_by_vector.return_value = [
        SearchResult(Document(content=f"Source {i}: café", embedding=[0.1, 0.2],
                              storage_key=f"documents/{i}.txt"), i / 10)
        for i in range(3)
    ]
    model = Mock()
    model.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Answer [1]"}}],
    }
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=factory))
    return dict(query=" question ", model_path=str(path),
                embedding_provider=provider, repository=repository), model, factory


@pytest.mark.unit
def test_infer_retrieves_and_grounds_answer(setup_inference):
    kwargs, model, factory = setup_inference
    result = infer(**kwargs)
    kwargs["embedding_provider"].embed.assert_called_once_with([QUERY_PREFIX + "question"])
    kwargs["repository"].search_by_vector.assert_called_once_with(embedding=[0.1, 0.2], limit=3)
    messages = model.create_chat_completion.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "untrusted reference data" in messages[0]["content"]
    sources = json.loads(messages[0]["content"].split("relevance):\n", 1)[1])
    assert sources == result["documents"]
    assert len(sources) == 3
    assert sources[2]["citation"] == 3
    assert sources[0]["storage_key"] == "documents/0.txt"
    assert messages[1] == {"role": "user", "content": "question"}
    assert result["response"] == "Answer [1]"
    factory.assert_called_once_with(model_path=kwargs["model_path"], n_ctx=4096,
                                    n_gpu_layers=0, verbose=False)
    model.close.assert_called_once()
    kwargs["repository"].create.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("key,value", [("query", " "), ("top_k", 0), ("top_k", True),
                                      ("context_size", 0), ("max_tokens", 4096)])
def test_invalid_input_does_not_load_model(setup_inference, key, value):
    kwargs, _, factory = setup_inference
    kwargs[key] = value
    with pytest.raises(ValueError):
        infer(**kwargs)
    factory.assert_not_called()
    kwargs["embedding_provider"].embed.assert_not_called()


@pytest.mark.unit
def test_missing_model(setup_inference):
    kwargs, _, factory = setup_inference
    kwargs["model_path"] += ".missing"
    with pytest.raises(FileNotFoundError, match="Generation model not found"):
        infer(**kwargs)
    factory.assert_not_called()


@pytest.mark.unit
def test_no_documents_skips_generation(setup_inference):
    kwargs, _, factory = setup_inference
    kwargs["repository"].search_by_vector.return_value = []
    result = infer(**kwargs)
    assert result["documents"] == []
    assert "No indexed documents" in result["response"]
    factory.assert_not_called()


@pytest.mark.unit
@pytest.mark.parametrize("failure", [ValueError("context window exceeded"), RuntimeError("decode failed")])
def test_generation_failure_closes_model(setup_inference, failure):
    kwargs, model, _ = setup_inference
    model.create_chat_completion.side_effect = failure
    with pytest.raises(type(failure), match=str(failure)):
        infer(**kwargs)
    model.close.assert_called_once()


@pytest.mark.unit
def test_empty_generation_is_error(setup_inference):
    kwargs, model, _ = setup_inference
    model.create_chat_completion.return_value["choices"][0]["message"]["content"] = None
    with pytest.raises(RuntimeError, match="empty response"):
        infer(**kwargs)
    model.close.assert_called_once()
