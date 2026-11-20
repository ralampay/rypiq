"""Real local model reuse without downloads or external services."""
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.application import embed_file
from app.providers.config import ProviderSettings
from app.providers.contracts import ChatMessage
from app.providers.session import LocalModelSession


def test_local_models_load_once_across_warmup_and_repeated_operations(tmp_path, monkeypatch):
    embedding_path = Path(os.getenv('EMBEDDING_MODEL_PATH', 'models/embeddinggemma-300m-Q4_0.gguf'))
    inference_path = os.getenv('INFERENCE_MODEL_PATH')
    if not embedding_path.is_file() or not inference_path or not Path(inference_path).is_file():
        pytest.skip('Set local embedding and INFERENCE_MODEL_PATH files for session reuse integration')
    import llama_cpp
    original = llama_cpp.Llama
    models = []
    def load(**kwargs):
        model = original(**kwargs)
        model.close = Mock(wraps=model.close)
        models.append(model)
        return model
    constructor = Mock(side_effect=load)
    monkeypatch.setattr(llama_cpp, 'Llama', constructor)
    source = tmp_path / 'input.txt'
    source.write_text('Local models can be reused.')
    with LocalModelSession() as session:
        session.warmup(
            ProviderSettings(provider='local', capability='embedding', model=str(embedding_path), embedding_size=768),
            ProviderSettings(provider='local', capability='inference', model=inference_path, embedding_size=768),
        )
        for _ in range(2):
            vectors = embed_file(path=str(source), chunk_size=100, stride=100,
                                 model_path=str(embedding_path), model_session=session)
            assert len(vectors['chunks'][0]['embedding']) == 768
            with session.open_provider(ProviderSettings(provider='local', capability='inference',
                    model=inference_path, embedding_size=768)) as provider:
                result = provider.chat([ChatMessage('user', 'Say hello.')],
                                       max_tokens=32, temperature=0)
            assert result.text.strip()
        assert constructor.call_count == 2
        assert [call.kwargs['embedding'] for call in constructor.call_args_list] == [True, False]
        for model in models:
            model.close.assert_not_called()
    for model in models:
        model.close.assert_called_once()
