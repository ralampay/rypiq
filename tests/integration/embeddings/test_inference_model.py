"""Opt-in real local inference; never download model weights."""
import os
from pathlib import Path

import pytest

from app.providers.config import ProviderSettings
from app.providers.contracts import ChatMessage
from app.providers.factory import open_provider


def test_real_local_inference():
    path = os.getenv('INFERENCE_MODEL_PATH')
    if not path or not Path(path).is_file():
        pytest.skip('Set INFERENCE_MODEL_PATH to an existing chat-capable GGUF for local inference integration')
    settings = ProviderSettings(provider='local', capability='inference', model=path, embedding_size=768)
    with open_provider(settings) as provider:
        result = provider.chat([ChatMessage('user', 'Say hello.')], max_tokens=16)
    assert isinstance(result.text, str) and result.text.strip()
