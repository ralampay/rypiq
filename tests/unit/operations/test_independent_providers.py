from dataclasses import replace
from itertools import product
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app import application, cli
from app.models.application import Application
from app.models.processing import embedding_settings
from app.providers.resolution import settings_for


@pytest.mark.parametrize('embedding,inference', list(product(('local', 'openai', 'llama-server'), repeat=2)))
def test_independent_provider_resolution(embedding, inference):
    value = Application(name='mixed', embedding_model='embed', inference_model='chat',
                        embedding_provision_type=embedding, inference_provision_type=inference,
                        data={'provider_config': {
                            'embedding': {'base_url': 'http://localhost:8001/v1', 'api_key_env': 'EMBED_KEY'},
                            'inference': {'base_url': 'http://localhost:8002/v1', 'api_key_env': 'CHAT_KEY'}}})
    assert settings_for(value, 'embedding').provider == embedding
    assert settings_for(value, 'inference').provider == inference
    assert settings_for(value, 'embedding').api_key_env == 'EMBED_KEY'
    assert settings_for(value, 'inference').base_url == 'http://localhost:8002/v1'
    assert embedding_settings(value) == embedding_settings(replace(value, inference_provision_type='local'))


def test_update_cli_preserves_embedding_and_data(monkeypatch, capsys):
    value = Application(name='mixed', embedding_model='embed', data={'keep': True})
    monkeypatch.setattr(application, 'show_application', lambda **kwargs: value)
    update = Mock(side_effect=lambda **kwargs: kwargs['application'])
    monkeypatch.setattr(application, 'update_application', update)
    cli.main(['applications:update', str(value.id), '--inference-provision-type', 'openai',
              '--inference-model', 'chat'])
    updated = update.call_args.kwargs['application']
    assert updated.embedding_provision_type == 'local'
    assert updated.inference_provision_type == 'openai'
    assert updated.data == {'keep': True} and updated.embedding_model == 'embed'
    assert value.inference_model is None


def test_defaults_are_independent():
    value = Application(name='mixed', embedding_model='embed', embedding_provision_type='openai')
    assert value.inference_provision_type == 'local'


def test_legacy_provider_flag_is_rejected():
    with pytest.raises(SystemExit) as error:
        cli.main(['applications:update', str(uuid4()), '--provision-type', 'openai'])
    assert error.value.code == 2


def test_update_data_replaces_and_clear_retains_provider(monkeypatch, capsys):
    value = Application(name='mixed', embedding_model='embed', inference_model='chat',
                        inference_provision_type='openai', data={'remove': True})
    monkeypatch.setattr(application, 'show_application', lambda **kwargs: value)
    update = Mock(side_effect=lambda **kwargs: kwargs['application'])
    monkeypatch.setattr(application, 'update_application', update)
    cli.main(['applications:update', str(value.id), '--clear-inference-model', '--data', '{"new": true}'])
    updated = update.call_args.kwargs['application']
    assert updated.data == {'new': True} and updated.inference_model is None
    assert updated.inference_provision_type == 'openai'


@pytest.mark.parametrize('embedding,inference', list(product(('local', 'openai', 'llama-server'), repeat=2)))
def test_mixed_chat_uses_sessions_only_for_local_capabilities(monkeypatch, embedding, inference):
    from contextlib import contextmanager
    from unittest.mock import MagicMock
    from app.providers.types.chat import ChatMessage, ChatResult
    value = Application(name='mixed', embedding_model='embed', inference_model='chat', embedding_size=2,
                        embedding_provision_type=embedding, inference_provision_type=inference,
                        data={'provider_config': {'embedding': {'base_url': 'http://localhost:8001/v1'},
                                                  'inference': {'base_url': 'http://localhost:8002/v1'}}})
    database = MagicMock()
    database.__enter__.return_value = database
    database.applications.return_value.get.return_value = value
    database.embeddings.return_value.search_by_vector.return_value = []
    monkeypatch.setattr(application, 'get_database', lambda config=None: database)
    embedder = Mock(model_name='embed', embed=Mock(return_value=[[1., 0.]]))
    inference_provider = Mock(chat=Mock(return_value=ChatResult('answer', 'chat', 'stop')))
    calls, closed = [], []
    def scope(kind):
        @contextmanager
        def open_provider(settings, *, credentials=None):
            calls.append((kind, settings.capability, settings.provider))
            try:
                yield embedder if settings.capability == 'embedding' else inference_provider
            finally:
                closed.append(settings.capability)
        return open_provider
    session = Mock(open_provider=scope('session'))
    monkeypatch.setattr(application, 'open_provider', scope('remote'))
    result = application.chat(messages=[ChatMessage('user', 'Question')], application_id=value.id, model_session=session)
    assert result.text == 'answer'
    assert calls == [('session' if embedding == 'local' else 'remote', 'embedding', embedding),
                     ('session' if inference == 'local' else 'remote', 'inference', inference)]
    assert closed == ['inference', 'embedding']


def test_update_normalizes_only_changed_local_model(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    monkeypatch.chdir(tmp_path)
    original = Application(name='existing', embedding_model='relative-embed.gguf',
                           inference_provision_type='openai', inference_model='remote')
    database = MagicMock()
    database.__enter__.return_value = database
    database.applications.return_value.get.return_value = original
    database.applications.return_value.update.side_effect = lambda value: value
    monkeypatch.setattr(application, 'get_database', lambda config=None: database)
    updated = application.update_application(application=replace(original, inference_provision_type='local',
                                                                 inference_model='chat.gguf'))
    assert updated.embedding_model == original.embedding_model
    assert updated.inference_model == str(tmp_path / 'chat.gguf')


@pytest.mark.parametrize('providers,expected', [
    (('local', 'openai'), ['openai']),
    (('openai', 'local'), ['openai']),
    (('openai', 'llama-server'), ['openai', 'llama-server']),
    (('openai', 'openai'), ['openai']),
])
def test_discovery_uses_each_remote_provider_and_deduplicates(monkeypatch, providers, expected):
    from contextlib import contextmanager
    from unittest.mock import MagicMock
    value = Application(name='mixed', embedding_model='embed', inference_model='chat',
                        embedding_provision_type=providers[0], inference_provision_type=providers[1],
                        data={'provider_config': {cap: {'base_url': 'http://localhost:8000/v1',
                                                        'api_key_env': 'SHARED_KEY'}
                                                  for cap in ('embedding', 'inference')}})
    database = MagicMock()
    database.__enter__.return_value = database
    database.applications.return_value.get.return_value = value
    monkeypatch.setattr(application, 'get_database', lambda config=None: database)
    from app.config import AppConfig
    monkeypatch.setattr(application, 'load_config', lambda: AppConfig(catalog={}))
    seen = []
    @contextmanager
    def discover(settings, *, credentials=None):
        seen.append(settings.provider)
        yield Mock(list_models=Mock(return_value=[]))
    monkeypatch.setattr(application, 'open_discovery_provider', discover)
    assert application.list_models(application_id=value.id, discover=True) == []
    assert seen == expected


def test_mixed_catalog_aliases_and_local_path_resolution(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    import yaml
    catalog = tmp_path / 'models.yml'
    catalog.write_text(yaml.safe_dump({'models': {
        'embed': {'provider': 'local', 'capabilities': ['embedding'], 'path': 'embed.gguf'},
        'chat': {'provider': 'openai', 'capabilities': ['inference'], 'model': 'remote-chat'},
    }}))
    monkeypatch.setenv('RYPIQ_MODELS_CONFIG', str(catalog))
    database = MagicMock()
    database.__enter__.return_value = database
    database.settings = {'dimensions': 768}
    database.applications.return_value.create.side_effect = lambda value: value
    monkeypatch.setattr(application, 'get_database', lambda config=None: database)
    value = application.create_application(name='mixed', embedding_model='@embed', inference_model='@chat',
                                           inference_provision_type='openai')
    assert value.embedding_model == str(tmp_path / 'embed.gguf')
    assert value.inference_model == 'remote-chat'
    assert value.embedding_provision_type == 'local'
