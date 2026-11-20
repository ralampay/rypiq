"""Explicit opt-in live tests. Configuration files contain environment references, not keys."""
import json
import os
from pathlib import Path

import pytest

from app.models.application import Application
from app.providers.config import settings_for
from app.providers.contracts import ChatMessage
from app.providers.factory import open_capability_provider as open_provider, open_discovery_provider
from app.providers.resolution import discovery_settings_for


@pytest.mark.parametrize('provider,variable', [
    ('openai', 'RYPIQ_LIVE_OPENAI_CONFIG'), ('llama-server', 'RYPIQ_LIVE_LLAMA_SERVER_CONFIG'),
])
def test_live_remote_embedding_inference_and_discovery(provider, variable):
    path = os.getenv(variable)
    if not path:
        pytest.skip(f'Set {variable} to explicitly opt in to live provider requests')
    options = json.loads(Path(path).read_text())
    application = Application(name='live integration', embedding_provision_type=provider, inference_provision_type=provider, **options)
    with open_provider(settings_for(application, 'embedding')) as embedding:
        vectors = embedding.embed(['Hello world'])
        assert len(vectors) == 1 and len(vectors[0]) == application.embedding_size
    with open_discovery_provider(discovery_settings_for(application, 'embedding')) as discovery:
        assert isinstance(discovery.list_models(), list)
    with open_provider(settings_for(application, 'inference')) as inference:
        result = inference.chat([ChatMessage('user', 'Say hello.')], max_tokens=32)
        assert isinstance(result.text, str)
