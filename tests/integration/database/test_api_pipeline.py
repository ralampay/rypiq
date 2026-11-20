from contextlib import contextmanager
from unittest.mock import Mock

from fastapi.testclient import TestClient

from app import application, create_app, worker
from app.config import AppConfig, ApiSettings, DatabaseSettings, StorageSettings
from app.logging.config import LoggingSettings
from app.providers.config import model_identity
from app.providers.types.chat import ChatResult
from app.storage.providers.local import LocalStorage


def test_http_upload_worker_poll_search_and_delete(backend, tmp_path, monkeypatch):
    backend.create_database()
    monkeypatch.setenv('APP_SECRET_KEY', 'ambient-key-must-not-win')
    storage = LocalStorage(tmp_path / 'objects')
    config = AppConfig(api=ApiSettings(secret_key='pipeline-test-key'),
                       database=DatabaseSettings(**backend.settings),
                       storage=StorageSettings(tmp_path / 'objects'),
                       logging=LoggingSettings(directory=tmp_path / 'logs'))
    closed = []
    @contextmanager
    def provider(settings, *args, **kwargs):
        value = Mock(model_name=model_identity(settings.model, settings.provider))
        value.embed.side_effect = lambda texts: [[1., 0.] for _ in texts]
        value.chat.return_value = ChatResult(text='grounded answer', model='fake', finish_reason='stop')
        try:
            yield value
        finally:
            closed.append(settings.capability)
    monkeypatch.setattr(application, '_open_configured_provider', provider)
    headers = {'X-App-Secret-Key': 'pipeline-test-key'}
    with TestClient(create_app(config=config)) as client:
        result = client.post('/api/v1/applications', headers=headers, json={
            'name': 'API test', 'embedding_model': 'fake', 'embedding_size': 2,
            'inference_model': 'fake-chat', 'chunk_size': 5, 'stride': 5})
        assert result.status_code == 201
        app_id = result.json()['id']
        response = client.post(f'/api/v1/applications/{app_id}/documents', headers=headers,
            data={'name': 'Report'}, files={'file': ('report.txt', b'abcdefghij')})
        assert response.status_code == 201
        document = response.json()
        assert document['status'] == 'pending'
        assert closed == []
        doc_id = document['id']
        queued = client.post(f'/api/v1/documents/{doc_id}/process', headers=headers)
        assert queued.status_code == 202
        assert client.delete(f'/api/v1/documents/{doc_id}', headers=headers).status_code == 409
        assert client.get(queued.headers['location'], headers=headers).json()['state'] == 'queued'
        assert worker.run_once(config=config)
        assert closed == ['embedding']
        assert client.get(queued.headers['location'], headers=headers).json()['state'] == 'succeeded'
        status = client.get(f'/api/v1/documents/{doc_id}/status', headers=headers).json()
        assert status['status'] == 'done'
        details = client.get(f'/api/v1/documents/{doc_id}', headers=headers).json()
        assert details['embedding_count'] == 2
        chunk = client.get('/api/v1/embeddings/' + details['embeddings'][0]['id'], headers=headers)
        assert chunk.status_code == 200
        assert chunk.json()['embedding'] == [1., 0.]
        search = client.post(f'/api/v1/applications/{app_id}/search', headers=headers, json={'query': 'test'})
        assert search.status_code == 200
        assert len(search.json()) == 2
        chat = client.post(f'/api/v1/applications/{app_id}/chat', headers=headers, json={'prompt': 'test'})
        assert chat.status_code == 200
        assert chat.json()['text'] == 'grounded answer'
        assert all('source' not in source for source in chat.json()['sources'])
        assert client.delete(f'/api/v1/documents/{doc_id}', headers=headers).json()['deleted']
        assert client.get(queued.headers['location'], headers=headers).status_code == 404
        assert storage.exists(document['storage_key'])
        assert client.delete(f'/api/v1/applications/{app_id}', headers=headers).json()['deleted']
