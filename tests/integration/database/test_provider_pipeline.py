"""Configured remote workflows across actual SDK, HTTP, storage and PostgreSQL."""
import pytest

from app import application
from app.providers.contracts import ChatMessage, ProviderError
from app.storage.providers.local import LocalStorage


@pytest.mark.parametrize('provider', ['openai', 'llama-server'])
def test_remote_pipeline_preserves_application_scope_and_storage_order(backend, server, tmp_path, monkeypatch, provider):
    backend.create_database()
    monkeypatch.setattr(application, 'get_database', lambda: backend)
    monkeypatch.setenv('TEST_PIPELINE_MODEL_KEY', 'test-only')
    storage = LocalStorage(tmp_path / 'objects')
    monkeypatch.setattr('app.storage.factory.get_storage', lambda: storage)
    config = {'provider_config': {
        'embedding': {'base_url': server['url'] + '/v1', 'api_key_env': 'TEST_PIPELINE_MODEL_KEY'},
        'inference': {'base_url': server['url'] + '/v1', 'api_key_env': 'TEST_PIPELINE_MODEL_KEY'},
    }}
    first = application.create_application(name='first', provision_type=provider, embedding_model='remote-model',
                                            embedding_size=2, inference_model='remote-model', data=config)
    second = application.create_application(name='second', provision_type=provider, embedding_model='remote-model',
                                             embedding_size=2, data=config)
    source = tmp_path / 'source.txt'
    source.write_text('hello world')
    docs = application.ingest_file(path=str(source), chunk_size=100, stride=100, application_id=first.id)
    assert len(docs.embeddings) == 1 and docs.document.application_id == first.id
    assert docs.embeddings[0].provision_type == provider and docs.embeddings[0].embedding_model == 'remote-model'
    assert (storage.root / docs.document.storage_key).read_text() == 'hello world'
    matches = application.search_documents(query='hello', application_id=first.id)
    assert [match.document.id for match in matches] == [docs.document.id]
    assert application.search_documents(query='hello', application_id=second.id) == []
    assert application.chat(messages=[ChatMessage('user', 'hello')], application_id=first.id).text == 'answer'
    before = len(server['requests'])
    assert any(m.model == 'remote-model' for m in application.list_models(application_id=first.id, discover=True))
    assert len(server['requests']) == before + 1  # Shared endpoint discovered once.
    wrong_size = application.create_application(name='wrong size', provision_type=provider,
                                                 embedding_model='remote-model', embedding_size=3, data=config)
    stored_before = [item.key for item in storage.list()]
    with pytest.raises(ProviderError, match='response'):
        application.ingest_file(path=str(source), chunk_size=100, stride=100, application_id=wrong_size.id)
    assert [item.key for item in storage.list()] == stored_before
    assert len(backend.documents().list()) == 1
