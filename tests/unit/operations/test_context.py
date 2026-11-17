from types import SimpleNamespace
from contextlib import contextmanager

import pytest

from app import context as context_module
from app import cli, application


def test_context_owns_and_releases_database(monkeypatch):
    calls = []
    backend = SimpleNamespace(documents=lambda: 'repository', close=lambda: calls.append('closed'))
    monkeypatch.setattr(context_module, 'get_database', lambda: backend)
    monkeypatch.setattr(context_module, 'get_storage', lambda: 'storage')
    context = context_module.create_context()
    assert context.documents == 'repository'
    context.close()
    assert calls == ['closed']


def test_context_setup_failure_releases_database(monkeypatch):
    calls = []
    backend = SimpleNamespace(documents=lambda: 'repository', close=lambda: calls.append('closed'))
    monkeypatch.setattr(context_module, 'get_database', lambda: backend)
    def fail():
        raise ValueError('bad storage configuration')
    monkeypatch.setattr(context_module, 'get_storage', fail)
    with pytest.raises(ValueError):
        context_module.create_context()
    assert calls == ['closed']


def test_search_cli_failure_releases_database(monkeypatch):
    calls = []
    context = context_module.ApplicationContext(documents='repository', storage='storage',
        database=SimpleNamespace(close=lambda: calls.append('closed')))
    @contextmanager
    def open_repository():
        with context:
            yield context.documents
    monkeypatch.setattr(application, 'open_document_repository', open_repository)
    def fail(**kwargs):
        raise ValueError('bad model')
    monkeypatch.setattr(application, 'get_embedding_provider', fail)
    monkeypatch.setattr('sys.argv', ['rypiq', 'search', 'hello'])
    with pytest.raises(SystemExit):
        cli.main()
    assert calls == ['closed']
