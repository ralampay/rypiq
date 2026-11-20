from unittest.mock import Mock
from uuid import uuid4

from fastapi.testclient import TestClient

from app import application, create_app


def test_default_cors_allows_secret_header_preflight_without_auth(monkeypatch):
    monkeypatch.setenv('APP_SECRET_KEY', 'test-secret')
    monkeypatch.delenv('API_CORS_ORIGINS', raising=False)
    with TestClient(create_app()) as client:
        response = client.options('/api/v1/applications', headers={
            'Origin': 'http://localhost:5173',
            'Access-Control-Request-Method': 'POST',
            'Access-Control-Request-Headers': 'content-type,x-app-secret-key',
        })
        assert response.status_code == 200
        assert response.headers['access-control-allow-origin'] == '*'
        assert 'x-app-secret-key' in response.headers['access-control-allow-headers'].lower()
        assert 'access-control-allow-credentials' not in response.headers
        assert client.options('/api/v1/applications').status_code == 401


def test_cors_headers_on_success_auth_and_unexpected_error(monkeypatch):
    monkeypatch.setenv('APP_SECRET_KEY', 'test-secret')
    monkeypatch.setenv('API_CORS_ORIGINS', 'http://localhost:5173')
    monkeypatch.setattr(application, 'show_application', Mock(side_effect=RuntimeError('private')))
    with TestClient(create_app()) as client:
        headers = {'Origin': 'http://localhost:5173'}
        unauthenticated = client.get('/api/v1/health', headers=headers)
        assert unauthenticated.status_code == 401
        headers['X-App-Secret-Key'] = 'test-secret'
        success = client.get('/api/v1/health', headers=headers)
        assert success.status_code == 200
        failure = client.get(f'/api/v1/applications/{uuid4()}', headers=headers)
        assert failure.status_code == 500
        for response in (unauthenticated, success, failure):
            assert response.headers['access-control-allow-origin'] == 'http://localhost:5173'
            assert 'Location' in response.headers['access-control-expose-headers']
            assert 'Origin' in response.headers['vary']


def test_configured_origins_and_rejected_preflight(monkeypatch):
    monkeypatch.setenv('APP_SECRET_KEY', 'test-secret')
    monkeypatch.setenv('API_CORS_ORIGINS', ' http://localhost:5173, https://frontend.example , ')
    with TestClient(create_app()) as client:
        for origin in ('http://localhost:5173', 'https://frontend.example'):
            response = client.options('/api/v1/documents/example/process', headers={
                'Origin': origin, 'Access-Control-Request-Method': 'POST',
                'Access-Control-Request-Headers': 'X-App-Secret-Key',
            })
            assert response.status_code == 200
            assert response.headers['access-control-allow-origin'] == origin
        for origin, method, requested_header in (
            ('https://other.example', 'POST', 'X-App-Secret-Key'),
            ('http://localhost:5173', 'TRACE', 'X-App-Secret-Key'),
            ('http://localhost:5173', 'POST', 'X-Unsupported'),
        ):
            response = client.options('/api/v1/applications', headers={
                'Origin': origin, 'Access-Control-Request-Method': method,
                'Access-Control-Request-Headers': requested_header,
            })
            assert response.status_code == 400


def test_empty_origin_list_disables_cross_origin_access(monkeypatch):
    monkeypatch.setenv('APP_SECRET_KEY', 'test-secret')
    monkeypatch.setenv('API_CORS_ORIGINS', '')
    with TestClient(create_app()) as client:
        response = client.get('/api/v1/health', headers={
            'Origin': 'http://localhost:5173', 'X-App-Secret-Key': 'test-secret',
        })
        assert response.status_code == 200
        assert 'access-control-allow-origin' not in response.headers
