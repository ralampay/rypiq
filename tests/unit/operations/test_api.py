from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import app
from app import application
from app.exceptions import ConfigurationError
from app.models.application import Application
from app.models.document import Document, DocumentDetails


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv('APP_SECRET_KEY', 'test-service-secret')
    factory = getattr(app, 'create_app', None)
    assert callable(factory), 'The API must expose an app factory'
    with TestClient(factory(), raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def headers():
    return {'X-App-Secret-Key': 'test-service-secret'}


def test_missing_secret_fails_closed(monkeypatch):
    monkeypatch.setenv('APP_SECRET_KEY', '')
    factory = getattr(app, 'create_app', None)
    assert callable(factory), 'The API must expose an app factory'
    with pytest.raises(ConfigurationError, match='APP_SECRET_KEY'):
        factory()


def test_auth_before_business_io(client, monkeypatch, headers):
    operation = Mock()
    monkeypatch.setattr(application, 'enumerate_application_summaries', operation)
    for candidate in ({}, {'X-App-Secret-Key': 'wrong'}):
        assert client.get('/api/v1/applications', headers=candidate).status_code == 401
    operation.assert_not_called()
    assert client.get('/api/v1/health', headers=headers).json() == {'status': 'ok'}
    assert client.get('/api/v1/doctor').status_code == 401


def test_create_application_omits_record_key(client, monkeypatch, headers):
    record = Application(name='demo', embedding_model='fake', embedding_size=3)
    operation = Mock(return_value=record)
    monkeypatch.setattr(application, 'create_application', operation)
    response = client.post('/api/v1/applications', headers=headers,
                           json={'name': 'demo', 'embedding_model': 'fake'})
    assert response.status_code == 201
    assert response.json()['id'] == str(record.id)
    assert 'api_key' not in response.json()


def test_invalid_request_and_safe_errors(client, monkeypatch, headers):
    operation = Mock(side_effect=RuntimeError('secret-database-url'))
    monkeypatch.setattr(application, 'show_application', operation)
    assert client.get('/api/v1/applications/not-a-uuid', headers=headers).status_code == 422
    operation.assert_not_called()
    response = client.get(f'/api/v1/applications/{uuid4()}', headers=headers)
    assert response.status_code == 500
    assert 'secret-database-url' not in response.text


def test_text_and_openapi(client, headers):
    response = client.post('/api/v1/processing/text', headers=headers, json={'content': 'hello world'})
    assert response.json()['words'] == 2
    schema = client.get('/openapi.json').json()
    assert schema['components']['securitySchemes']['AppSecret']['name'] == 'X-App-Secret-Key'
    assert schema['paths']['/api/v1/documents/{document_id}/process']['post']['security']


def test_upload_stores_without_embedding_and_cleans_temp(client, monkeypatch, headers):
    seen = []
    def store(**kwargs):
        path = Path(kwargs['path'])
        assert path.read_bytes() == b'hello'
        seen.append(path)
        return DocumentDetails(Document(name=kwargs['name'], application_id=kwargs['application_id'],
                                        filename=path.name, source=str(path)), [])
    monkeypatch.setattr(application, 'store_document', store)
    process = Mock()
    monkeypatch.setattr(application, 'process_document', process)
    response = client.post(f'/api/v1/applications/{uuid4()}/documents', headers=headers,
                           data={'name': 'demo'}, files={'file': ('../../demo.txt', b'hello')})
    assert response.status_code == 201
    assert response.json()['status'] == 'pending'
    assert 'source' not in response.json()
    assert response.json()['filename'] == 'demo.txt'
    assert not seen[0].exists()
    process.assert_not_called()


def test_text_file_returns_chunk_offsets_without_temporary_path(client, headers):
    response = client.post('/api/v1/processing/text-file', headers=headers,
                           data={'chunk_size': 5, 'stride': 5}, files={'file': ('demo.txt', b'abcdefghij')})
    assert response.status_code == 200
    assert response.json()['chunks'] == [
        {'index': 0, 'start': 0, 'end': 5, 'text': 'abcde'},
        {'index': 1, 'start': 5, 'end': 10, 'text': 'fghij'}]
    assert 'source' not in response.json()


def test_upload_limits_and_failure_cleanup(client, monkeypatch, headers):
    seen = []
    def fail(**kwargs):
        seen.append(Path(kwargs['path']))
        raise RuntimeError('private-upload-error')
    monkeypatch.setattr(application, 'store_document', fail)
    url = f'/api/v1/applications/{uuid4()}/documents'
    response = client.post(url, headers=headers, data={'name': 'demo'}, files={'file': ('x.txt', b'hello')})
    assert response.status_code == 500
    assert not seen[0].exists()
    assert 'private-upload-error' not in response.text
    from app.dependencies.config import get_config
    settings = client.app.state.config
    client.app.dependency_overrides[get_config] = lambda: replace(
        settings, api=replace(settings.api, max_upload_bytes=2))
    response = client.post(url, headers=headers, data={'name': 'demo'}, files={'file': ('x.txt', b'hello')})
    assert response.status_code == 413
    assert len(seen) == 1


def test_update_null_semantics(client, monkeypatch, headers):
    original = Application(name='demo', embedding_model='fake', inference_model='chat')
    monkeypatch.setattr(application, 'show_application', lambda **_: original)
    operation = Mock(side_effect=lambda **kwargs: kwargs['application'])
    monkeypatch.setattr(application, 'update_application', operation)
    url = f'/api/v1/applications/{original.id}'
    assert client.patch(url, headers=headers, json={'name': 'renamed'}).json()['inference_model'] == 'chat'
    assert client.patch(url, headers=headers, json={'inference_model': None}).json()['inference_model'] is None
    assert client.patch(url, headers=headers, json={'data': None}).status_code == 422


def test_queue_response_and_polling_exclude_claim_token(client, monkeypatch, headers):
    from datetime import datetime, timezone
    from app.models.processing_job import ProcessingJob
    job = ProcessingJob(id=uuid4(), document_id=uuid4(), batch_size=1, state='queued', attempts=0,
                        created_at=datetime.now(timezone.utc), updated_at=datetime.now(timezone.utc), claim_token=uuid4())
    monkeypatch.setattr(application, 'enqueue_document', lambda **_: job)
    monkeypatch.setattr(application, 'show_processing_job', lambda **_: job)
    response = client.post(f'/api/v1/documents/{job.document_id}/process', headers=headers)
    assert response.status_code == 202
    assert response.json()['job_id'] == str(job.id)
    polled = client.get(response.headers['location'], headers=headers)
    assert polled.status_code == 200
    assert 'claim_token' not in polled.json()
    assert polled.json()['state'] == 'queued'


def test_doctor_report_status(client, monkeypatch, headers):
    from app.controllers import system_controller
    from app.models.diagnostics import ConfigurationCheck, ConfigurationReport
    monkeypatch.setattr(system_controller, 'check_configuration', lambda config=None: ConfigurationReport('test', (
        ConfigurationCheck('Database', 'REQUIRED', 'Run db:migrate'),)))
    response = client.get('/api/v1/doctor', headers=headers)
    assert response.status_code == 503
    assert response.json()['incomplete_count'] == 1


def test_api_store_uses_original_filename_as_source(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from app.operations.documents.create import CreateDocument
    record = Application(name='demo', embedding_model='fake')
    database = Mock()
    database.applications.return_value.get.return_value = record
    repository = database.documents.return_value
    repository.name_exists.return_value = False
    repository.create.side_effect = lambda value: value
    repository.update.side_effect = lambda value: value
    storage = Mock()
    from app.storage.models import StoredObject
    storage.save.return_value = StoredObject('stored/file', 5, 'text/plain', 'checksum')
    monkeypatch.setattr(application, 'get_database', lambda config=None: nullcontext(database))
    monkeypatch.setattr('app.storage.factory.get_storage', lambda config=None: storage)
    path = tmp_path / 'report.txt'
    path.write_text('hello')
    result = application.store_document(name='Report', path=str(path), application_id=record.id)
    assert result.document.source == 'report.txt'


@pytest.mark.parametrize('error_name,code', [
    ('DocumentNotFoundError', 404), ('ConflictError', 409), ('ValidationError', 422),
    ('ProviderRequestError', 502), ('DatabaseConnectionError', 503), ('StorageError', 503),
])
def test_domain_errors_have_safe_http_status(client, monkeypatch, headers, error_name, code):
    from app import exceptions
    error = getattr(exceptions, error_name)('private credential text')
    monkeypatch.setattr(application, 'show_application', Mock(side_effect=error))
    response = client.get(f'/api/v1/applications/{uuid4()}', headers=headers)
    assert response.status_code == code
    assert 'private credential' not in response.text


def test_storage_upload_and_list(client, monkeypatch, headers, tmp_path):
    from app.controllers import storage_controller
    from app.storage.providers.local import LocalStorage
    storage = LocalStorage(tmp_path / 'objects')
    monkeypatch.setattr(storage_controller, 'get_storage', lambda config=None: storage)
    response = client.post('/api/v1/storage', headers=headers, data={'prefix': 'reports'},
                           files={'file': ('report.txt', b'example')})
    assert response.status_code == 201
    item = response.json()
    assert item['size'] == 7
    listed = client.get('/api/v1/storage?prefix=reports', headers=headers)
    assert len(listed.json()) == 1
    assert listed.json()[0]["key"] == item["key"]
    assert listed.json()[0]["size"] == item["size"]


def test_models_applications_documents_and_delete_shapes(client, monkeypatch, headers):
    from app.models.application import ApplicationSummary
    from app.models.document import DocumentSummary
    from app.providers.types.model import ModelInfo
    app_id, doc_id = uuid4(), uuid4()
    monkeypatch.setattr(application, 'enumerate_application_summaries', lambda config=None: [ApplicationSummary(
        id=app_id, name='demo', embedding_model='fake', embedding_size=3, inference_model=None,
        embedding_provision_type='local', inference_provision_type='local', document_count=1)])
    monkeypatch.setattr(application, 'enumerate_documents', lambda **_: [DocumentSummary(doc_id, 'doc', 2, 'done')])
    monkeypatch.setattr(application, 'list_models', lambda **_: [ModelInfo('fake', 'local', ('embedding',), 'catalog', 'available', 'fake')])
    monkeypatch.setattr(application, 'delete_embedding', lambda **_: False)
    assert client.get('/api/v1/applications', headers=headers).json()[0]['document_count'] == 1
    assert client.get(f'/api/v1/applications/{app_id}/documents?status=done', headers=headers).json()[0]['id'] == str(doc_id)
    assert client.get('/api/v1/models', headers=headers).json()[0]['capabilities'] == ['embedding']
    assert client.delete(f'/api/v1/embeddings/{uuid4()}', headers=headers).json() == {'deleted': False}


def test_duplicate_credentials_and_oversized_stream_fail_before_operation(monkeypatch):
    monkeypatch.setenv('APP_SECRET_KEY', 'test-key')
    monkeypatch.setenv('API_MAX_UPLOAD_BYTES', '1')
    operation = Mock()
    monkeypatch.setattr(application, 'store_document', operation)
    with TestClient(app.create_app()) as client:
        assert client.get('/api/v1/health', headers=[('X-App-Secret-Key', 'test-key'),
                                                   ('X-App-Secret-Key', 'test-key')]).status_code == 401
        response = client.post(f'/api/v1/applications/{uuid4()}/documents',
                               headers={'X-App-Secret-Key': 'test-key'}, data={'name': 'doc'},
                               files={'file': ('x.txt', b'x' * (1024 * 1024 + 100))})
        assert response.status_code == 413
    operation.assert_not_called()


def test_all_service_routes_declare_auth(client):
    schema = client.get('/openapi.json').json()
    for path, methods in schema['paths'].items():
        assert path.startswith('/api/v1/')
        for operation in methods.values():
            assert operation['security'] == [{'AppSecret': []}]


def test_unexpected_failure_does_not_escape_to_server_logging(monkeypatch, caplog):
    monkeypatch.setenv('APP_SECRET_KEY', 'test-key')
    monkeypatch.setattr(application, 'show_application', Mock(side_effect=RuntimeError('private-driver-url')))
    with TestClient(app.create_app(), raise_server_exceptions=True) as client:
        response = client.get(f'/api/v1/applications/{uuid4()}', headers={'X-App-Secret-Key': 'test-key'})
        assert response.status_code == 500
    assert 'private-driver-url' not in caplog.text
