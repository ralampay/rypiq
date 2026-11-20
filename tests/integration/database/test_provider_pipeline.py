"""Configured remote workflows across actual SDK, HTTP, storage and PostgreSQL."""
import pytest

from app import application
from app.providers.contracts import ChatMessage, ProviderError
from app.storage.providers.local import LocalStorage


@pytest.mark.parametrize('provider', ['openai', 'llama-server'])
@pytest.mark.parametrize('inference_provider', ['openai', 'llama-server'])
def test_remote_pipeline_preserves_application_scope_and_storage_order(backend, server, tmp_path, monkeypatch, provider, inference_provider, capsys):
    backend.create_database()
    catalog = tmp_path / 'models.yml'
    catalog.write_text('models: {}\n')
    monkeypatch.setenv('RYPIQ_MODELS_CONFIG', str(catalog))
    monkeypatch.setattr(application, 'get_database', lambda config=None: backend)
    monkeypatch.setenv('TEST_PIPELINE_MODEL_KEY', 'test-only')
    storage = LocalStorage(tmp_path / 'objects')
    monkeypatch.setattr('app.storage.factory.get_storage', lambda config=None: storage)
    config = {'provider_config': {
        'embedding': {'base_url': server['url'] + '/v1', 'api_key_env': 'TEST_PIPELINE_MODEL_KEY'},
        'inference': {'base_url': server['url'] + '/v1', 'api_key_env': 'TEST_PIPELINE_MODEL_KEY'},
    }}
    first = application.create_application(name='first', embedding_provision_type=provider, inference_provision_type=inference_provider, embedding_model='remote-model',
                                            embedding_size=2, inference_model='remote-model', data=config)
    second = application.create_application(name='second', embedding_provision_type=provider, inference_provision_type=provider, embedding_model='remote-model',
                                             embedding_size=2, data=config)
    source = tmp_path / 'source.txt'
    source.write_text('hello world')
    docs = application.ingest_file(path=str(source), chunk_size=100, stride=100, application_id=first.id, name='test document')
    assert len(docs.embeddings) == 1 and docs.document.application_id == first.id
    assert docs.embeddings[0].provision_type == provider and docs.embeddings[0].embedding_model == 'remote-model'
    assert (storage.root / docs.document.storage_key).read_text() == 'hello world'
    matches = application.search_documents(query='hello', application_id=first.id)
    assert [match.document.id for match in matches] == [docs.document.id]
    assert application.search_documents(query='hello', application_id=second.id) == []
    assert application.chat(messages=[ChatMessage('user', 'hello')], application_id=first.id).text == 'answer'
    path, request, _ = server['requests'][-1]
    assert path == '/v1/chat/completions'
    assert request['messages'][0]['role'] == 'system'
    assert 'insufficient evidence' in request['messages'][0]['content']
    assert 'hello world' in request['messages'][-1]['content']
    assert str(docs.embeddings[0].id) in request['messages'][-1]['content']
    before = len(server['requests'])
    assert any(m.model == 'remote-model' for m in application.list_models(application_id=first.id, discover=True))
    assert len(server['requests']) == before + (1 if provider == inference_provider else 2)
    wrong_size = application.create_application(name='wrong size', embedding_provision_type=provider, inference_provision_type=provider,
                                                 embedding_model='remote-model', embedding_size=3, data=config)
    stored_before = [item.key for item in storage.list()]
    with pytest.raises(ProviderError, match='response'):
        application.ingest_file(path=str(source), chunk_size=100, stride=100, application_id=wrong_size.id, name='test document')
    assert len(storage.list()) == len(stored_before) + 1
    assert len(backend.documents().list()) == 2
    assert backend.documents().list(status='error')[0].application_id == wrong_size.id

    # The CLI reuses the stored original after upload removal and picks up saved chunking changes.
    from dataclasses import replace
    import json
    from app import cli
    application.update_application(application=replace(first, data={**config,
        'chunking': {'chunk_size': 5, 'stride': 5}}))
    source.unlink()
    cli.main(['documents:process', '--document-id', str(docs.document.id)])
    processed = json.loads(capsys.readouterr().out)
    assert processed['id'] == str(docs.document.id)
    assert processed['status'] == 'done' and processed['embedding_count'] == 3
