from dataclasses import replace

import pytest
import yaml

from app.models.application import Application
from app.providers.config import settings_for
from app.providers.catalog import load_catalog, resolve_application


def test_legacy_local_defaults_and_remote_plain_inputs():
    local = Application(name='a', embedding_model='model.gguf')
    settings = settings_for(local, 'embedding')
    assert settings.model == 'model.gguf'
    assert settings.document_prefix == 'title: none | text: '
    assert settings.query_prefix == 'task: search result | query: '
    remote = settings_for(replace(local, provision_type='openai'), 'embedding')
    assert remote.base_url == 'https://api.openai.com/v1'
    assert remote.api_key_env == 'OPENAI_API_KEY'
    assert remote.document_prefix == remote.query_prefix == ''
    assert remote.timeout == 60


def test_separate_servers_and_dimension_configuration():
    app = Application(name='a', provision_type='llama-server', embedding_model='embed', inference_model='chat',
                      embedding_size=3, data={'provider_config': {
                          'embedding': {'base_url': 'http://localhost:8001/v1', 'dimensions': 3},
                          'inference': {'base_url': 'http://localhost:8002/v1', 'api_key_env': 'LLAMA_KEY'},
                      }})
    assert settings_for(app, 'embedding').dimensions == 3
    assert settings_for(app, 'inference').base_url == 'http://localhost:8002/v1'
    assert settings_for(app, 'inference').api_key_env == 'LLAMA_KEY'


@pytest.mark.parametrize('options', [
    {'api_key': 'never-store-secrets'}, {'timeout': 0}, {'timeout': True},
    {'base_url': 'file:///tmp/model'}, {'base_url': 'http://user:secret@host/v1'},
    {'base_url': 'http://host/v1?token=secret'}, {'dimensions': 4},
    {'context_size': 0}, {'gpu_layers': -2}, {'query_prefix': 1}, {'unknown': 1},
])
def test_invalid_options_are_rejected_without_echoing_values(options):
    app = Application(name='a', embedding_model='model', embedding_size=3,
                      data={'provider_config': {'embedding': options}})
    with pytest.raises(ValueError) as caught:
        settings_for(app, 'embedding')
    assert 'secret' not in str(caught.value)


def test_missing_inference_and_server_location():
    app = Application(name='a', embedding_model='embed', provision_type='llama-server')
    with pytest.raises(ValueError):
        settings_for(app, 'embedding')
    with pytest.raises(ValueError):
        settings_for(app, 'inference')


def test_catalog_resolution_snapshots_defaults_and_preserves_input(tmp_path):
    path = tmp_path / 'models.yml'
    path.write_text(yaml.safe_dump({'models': {'small': {
        'provider': 'local', 'capabilities': ['embedding'], 'path': 'weights.gguf',
        'options': {'document_prefix': '', 'query_prefix': '', 'context_size': 4096},
    }}}))
    catalog = load_catalog(path)
    original = Application(name='a', embedding_model='@small',
                           data={'other': True, 'provider_config': {'embedding': {'context_size': 1024}}})
    result = resolve_application(original, catalog)
    assert result.embedding_model == str(tmp_path / 'weights.gguf')
    assert result.data['provider_config']['embedding']['context_size'] == 1024
    assert result.data['provider_config']['embedding']['query_prefix'] == ''
    assert original.embedding_model == '@small'
    assert original.data['other'] is True
    path.write_text('models: {}')
    assert resolve_application(result, load_catalog(path)).embedding_model == result.embedding_model


@pytest.mark.parametrize('model,provider,capabilities', [('@missing', 'local', ['embedding']),
    ('@small', 'openai', ['embedding']), ('@small', 'local', ['inference'])])
def test_catalog_rejects_missing_or_incompatible_aliases(tmp_path, model, provider, capabilities):
    path = tmp_path / 'models.yml'
    path.write_text(yaml.safe_dump({'models': {'small': {
        'provider': provider, 'capabilities': capabilities,
        **({'path': 'model.gguf'} if provider == 'local' else {'model': 'remote'}),
    }}}))
    with pytest.raises(ValueError):
        resolve_application(Application(name='a', embedding_model=model), load_catalog(path))


@pytest.mark.parametrize('data', [[], None, 'invalid'])
def test_invalid_application_data_is_a_validation_error(data):
    with pytest.raises(ValueError):
        resolve_application(Application(name='a', embedding_model='model', data=data), {})
    with pytest.raises(ValueError):
        settings_for(Application(name='a', embedding_model='model', data=data), 'embedding')


def test_local_model_identity_matches_provider_for_symlinks(tmp_path):
    from app.providers.config import model_identity
    from app.application import _application_settings

    target = tmp_path / 'weights.gguf'
    target.touch()
    link = tmp_path / 'alias.gguf'
    link.symlink_to(target)
    assert model_identity(str(link), 'local') == target.name
    app = Application(name='a', embedding_model=str(target))
    assert _application_settings(app, 'embedding', str(link)).model == str(link)
    assert model_identity('org/model', 'openai') == 'org/model'
