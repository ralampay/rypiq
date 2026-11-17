from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from app.models.application import Application
from app.providers.catalog import load_catalog, resolve_application, catalog_models
from app.providers.config import ProviderSettings, settings_for
from app.providers.contracts import ChatMessage, ProviderError, ModelInfo
from app.providers.discovery import merge_models
from app.providers.factory import open_provider, create_provider
from app.providers.local import LocalInferenceProvider
from app.providers.remote import OpenAIProvider, LlamaServerProvider
from app.providers.normalization import validate_chat, validate_texts, embedding_response, chat_response


def setting(**values):
    return ProviderSettings(**{'provider': 'local', 'capability': 'inference', 'model': 'model',
                               'embedding_size': 2, **values})


@pytest.mark.parametrize('messages,tokens,temperature', [
    ([], 1, None), ([{}], 1, None), ([ChatMessage('tool', 'x')], 1, None),
    ([ChatMessage('user', '')], 1, None), ([ChatMessage('user', 'x')], 0, None),
    ([ChatMessage('user', 'x')], True, None), ([ChatMessage('user', 'x')], 1, True),
    ([ChatMessage('user', 'x')], 1, float('nan')), ([ChatMessage('user', 'x')], 1, 3),
])
def test_chat_validation(messages, tokens, temperature):
    with pytest.raises(ValueError):
        validate_chat(messages, tokens, temperature)


@pytest.mark.parametrize('texts', ['text', [''], [1], [' ']])
def test_invalid_embedding_inputs(texts):
    with pytest.raises(ValueError):
        validate_texts(texts)


def test_sdk_objects_and_response_validation():
    response = Mock()
    response.model_dump.return_value = {'data': [{'index': 0, 'embedding': [1., 0.]}]}
    assert embedding_response(response, 1, 2) == [[1., 0.]]
    response.model_dump.return_value = {'choices': [{'message': {'content': 'yes'}}]}
    assert chat_response(response, 'fallback').model == 'fallback'
    assert chat_response(response, 'fallback').usage is None
    with pytest.raises(ProviderError):
        embedding_response({'data': [{'index': False, 'embedding': [1., 0.]}]}, 1, 2)


def test_local_inference_parameters_cleanup_and_failures(tmp_path, monkeypatch):
    path = tmp_path / 'model.gguf'
    path.write_bytes(b'fake; never loaded')
    factory = Mock()
    factory.return_value.create_chat_completion.return_value = {
        'choices': [{'message': {'content': 'answer'}, 'finish_reason': 'stop'}]}
    value = setting(model=str(path), chat_format='chatml')
    adapter = LocalInferenceProvider(value, model_factory=factory)
    assert factory.call_args.kwargs['embedding'] is False
    assert factory.call_args.kwargs['chat_format'] == 'chatml'
    assert adapter.chat([ChatMessage('user', 'hello')], temperature=0).text == 'answer'
    adapter.chat([ChatMessage('system', 'brief'), ChatMessage('user', 'hi')])
    assert 'temperature' not in factory.return_value.create_chat_completion.call_args.kwargs
    factory.return_value.create_chat_completion.side_effect = RuntimeError('secret')
    with pytest.raises(ProviderError, match='request failed'):
        adapter.chat([ChatMessage('user', 'hello')])
    adapter.close()
    factory.return_value.close.assert_called_once()
    factory.return_value.close.side_effect = RuntimeError('secret')
    with pytest.raises(ProviderError, match='cleanup'):
        adapter.close()
    factory.side_effect = RuntimeError('secret')
    with pytest.raises(ProviderError, match='loaded'):
        LocalInferenceProvider(value, model_factory=factory)
    with pytest.raises(ValueError, match='does not exist'):
        LocalInferenceProvider(setting(), model_factory=factory)
    # Exercise the actual default factory selection without loading model weights.
    import llama_cpp
    fake = Mock()
    monkeypatch.setattr(llama_cpp, 'Llama', fake)
    LocalInferenceProvider(replace(value, chat_format=None)).close()
    fake.return_value.close.assert_called_once()


def test_remote_errors_capabilities_and_ownership():
    client = Mock()
    embedding = OpenAIProvider(setting(provider='openai', capability='embedding'), client=client)
    with pytest.raises(ValueError):
        embedding.chat([ChatMessage('user', 'hello')])
    inference = LlamaServerProvider(setting(provider='llama-server'), client=client, owns_client=True)
    with pytest.raises(ValueError):
        inference.embed(['hello'])
    client.chat.completions.create.return_value = {'choices': [{'message': {'content': 'yes'}}]}
    inference.chat([ChatMessage('user', 'hello')], temperature=0.5)
    assert client.chat.completions.create.call_args.kwargs['max_tokens'] == 512
    client.chat.completions.create.side_effect = RuntimeError('secret')
    with pytest.raises(ProviderError):
        inference.chat([ChatMessage('user', 'hello')])
    client.models.list.return_value = {'data': [{'id': ''}]}
    with pytest.raises(ProviderError):
        inference.list_models()
    client.models.list.return_value = [SimpleNamespace(model_dump=lambda: {'id': 'a'})]
    assert inference.list_models()[0].id == 'a'
    embedding.close()
    client.close.assert_not_called()
    client.close.side_effect = RuntimeError('secret')
    with pytest.raises(ProviderError):
        inference.close()


def test_provider_ownership_preserves_primary_failure(monkeypatch):
    model = Mock()
    monkeypatch.setattr('app.providers.factory.create_provider', lambda *args, **kwargs: model)
    with open_provider(setting()) as result:
        assert result is model
    model.close.assert_called_once()
    model.close.side_effect = RuntimeError('cleanup')
    with pytest.raises(ProviderError, match='completed'):
        with open_provider(setting()):
            pass
    failure = ValueError('primary')
    with pytest.raises(ValueError) as caught:
        with open_provider(setting()):
            raise failure
    assert caught.value is failure
    assert 'cleanup' in failure.__notes__[0]
    model.close.side_effect = None
    with pytest.raises(ValueError):
        with open_provider(setting()):
            raise failure


def test_factory_defaults_and_invalid_settings(monkeypatch):
    with pytest.raises(ValueError):
        create_provider(setting(provider='unknown'))
    with pytest.raises(ValueError):
        create_provider(setting(capability='unknown'))
    with pytest.raises(ValueError):
        create_provider(setting(provider='openai'))
    factory = Mock(side_effect=RuntimeError('secret'))
    with pytest.raises(ProviderError):
        create_provider(setting(provider='llama-server', base_url='http://localhost:1234/v1'), client_factory=factory)
    factory = Mock()
    create_provider(setting(provider='llama-server', base_url='http://localhost:1234/v1'), client_factory=factory).close()
    assert factory.call_args.kwargs['api_key'] == 'no-key-required'
    import openai
    monkeypatch.setattr(openai, 'OpenAI', factory)
    create_provider(setting(provider='llama-server', base_url='http://localhost:1234/v1')).close()
    local = Mock()
    monkeypatch.setattr('app.providers.factory.get_embedding_provider', local)
    assert create_provider(setting(capability='embedding')) is local.return_value
    monkeypatch.setattr('app.providers.factory.LocalInferenceProvider', local)
    assert create_provider(setting()) is local.return_value


@pytest.mark.parametrize('raw', [None, [], {'models': []}, {'models': {'@invalid': {}}},
    {'models': {'a': {'provider': 'invalid', 'capabilities': ['embedding']}}},
    {'models': {'a': {'provider': 'local', 'capabilities': []}}},
    {'models': {'a': {'provider': 'local', 'capabilities': ['embedding', 'embedding']}}},
    {'models': {'a': {'provider': 'local', 'capabilities': ['unknown']}}},
    {'models': {'a': {'provider': 'local', 'capabilities': ['embedding']}}}])
def test_invalid_catalog_shapes(tmp_path, raw):
    path = tmp_path / 'models.yml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        load_catalog(path)


def test_missing_invalid_catalog_and_environment_override(tmp_path, monkeypatch):
    path = tmp_path / 'catalog.yml'
    monkeypatch.setenv('RYPIQ_MODELS_CONFIG', str(path))
    with pytest.raises(ValueError):
        load_catalog()
    path.write_text('models: [broken')
    with pytest.raises(ValueError):
        load_catalog()
    path.write_text('models: {}')
    assert load_catalog() == {}


@pytest.mark.parametrize('data', [{'provider_config': []}, {'provider_config': {'unknown': {}}},
    {'provider_config': {'embedding': {'api_key_env': 'bad-name'}}},
    {'provider_config': {'inference': {'chat_format': 1}}},
    {'provider_config': {'embedding': {'base_url': 1}}}])
def test_invalid_settings_shapes(data):
    with pytest.raises(ValueError):
        settings_for(Application(name='a', embedding_model='x', data=data), 'embedding')


def test_unknown_capability():
    with pytest.raises(ValueError):
        settings_for(Application(name='a', embedding_model='x'), 'other')


def test_catalog_listing_and_discovery_merging(tmp_path):
    path = tmp_path / 'models.yml'
    (tmp_path / 'present.gguf').write_bytes(b'fake')
    path.write_text(yaml.safe_dump({'models': {
        'a': {'provider': 'local', 'capabilities': ['embedding'], 'path': 'present.gguf'},
        'b': {'provider': 'local', 'capabilities': ['inference'], 'path': 'missing.gguf'},
        'c': {'provider': 'openai', 'capabilities': ['inference'], 'model': 'remote'},
    }}))
    configured = catalog_models(load_catalog(path))
    assert [m.availability for m in configured] == ['available', 'missing', 'unverified']
    duplicate = replace(configured[2], id='@alias', capabilities=('embedding',))
    remote = ModelInfo('remote', 'openai', (), 'discovered', 'available', 'remote')
    unknown = replace(remote, id='unknown', model='unknown')
    merged = merge_models(configured + [duplicate], [remote, unknown])
    found = next(m for m in merged if m.model == 'remote')
    assert found.source == 'catalog+discovered'
    assert found.capabilities == ('embedding', 'inference')
    assert len(merged) == 4


@pytest.mark.parametrize('usage', [[], '', 1])
def test_chat_rejects_nonobject_usage(usage):
    with pytest.raises(ProviderError):
        chat_response({'choices': [{'message': {'content': 'yes'}}], 'usage': usage}, 'model')
