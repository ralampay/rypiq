from types import SimpleNamespace
from contextlib import contextmanager

import pytest
from unittest.mock import Mock
from app.models.application import Application

from app import context as context_module
from app import cli, application


def test_context_owns_and_releases_database(monkeypatch):
    calls = []
    backend = SimpleNamespace(documents=lambda: 'repository', close=lambda: calls.append('closed'))
    monkeypatch.setattr(context_module, 'get_database', lambda config=None: backend)
    monkeypatch.setattr(context_module, 'get_storage', lambda config=None: 'storage')
    context = context_module.create_context()
    assert context.documents == 'repository'
    context.close()
    assert calls == ['closed']


def test_context_setup_failure_releases_database(monkeypatch):
    calls = []
    backend = SimpleNamespace(documents=lambda: 'repository', close=lambda: calls.append('closed'))
    monkeypatch.setattr(context_module, 'get_database', lambda config=None: backend)
    def fail(config=None):
        raise ValueError('bad storage configuration')
    monkeypatch.setattr(context_module, 'get_storage', fail)
    with pytest.raises(ValueError):
        context_module.create_context()
    assert calls == ['closed']


def test_search_cli_failure_releases_database(monkeypatch):
    calls = []
    value = Application(name='test', embedding_model='fake', embedding_size=3)
    backend = Mock()
    backend.applications.return_value.get.return_value = value
    @contextmanager
    def open_backend(config=None):
        try:
            yield backend
        finally:
            calls.append('closed')
    monkeypatch.setattr(application, 'get_database', open_backend)
    def fail(**kwargs):
        raise ValueError('bad model')
    monkeypatch.setattr(application, 'get_embedding_provider', fail)
    monkeypatch.setattr('sys.argv', ['rypiq', 'search', 'hello', '--application-id', str(value.id)])
    with pytest.raises(SystemExit):
        cli.main()
    assert calls == ['closed']
