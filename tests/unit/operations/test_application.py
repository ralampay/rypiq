from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app import application, db
from app.context import ApplicationContext


def test_search_does_not_require_object_storage_configuration(monkeypatch):
    backend = Mock()
    backend.documents.return_value.search_by_vector.return_value = ['match']
    backend.__enter__ = Mock(return_value=backend)
    backend.__exit__ = Mock()
    monkeypatch.setattr(db, 'get_database', lambda: backend)
    def wrong_dependency():
        raise AssertionError('Search must not construct object storage')
    monkeypatch.setattr(application, 'create_context', wrong_dependency)
    provider = Mock()
    provider.embed.return_value = [[1., 0., 0.]]
    monkeypatch.setattr(application, 'get_embedding_provider', lambda **kwargs: provider)
    assert application.search_documents(query='hello') == ['match']
    provider.close.assert_called_once()
    backend.__exit__.assert_called_once()


def test_provider_cleanup_does_not_mask_primary_failure():
    model = Mock()
    model.close.side_effect = OSError('cleanup detail')
    primary = ValueError('primary')
    with pytest.raises(ValueError, match='primary') as error:
        with application.open_embedding_provider(provider_factory=lambda **kwargs: model):
            raise primary
    assert error.value is primary
    assert 'cleanup' in primary.__notes__[0]


def test_provider_cleanup_failure_after_success_is_reported():
    model = Mock()
    model.close.side_effect = OSError('cleanup detail')
    with pytest.raises(RuntimeError, match='completed.*cleanup'):
        with application.open_embedding_provider(provider_factory=lambda **kwargs: model):
            pass


def test_structural_provider_without_close_is_supported():
    model = SimpleNamespace(model_name='fake', embed=lambda texts: [])
    with application.open_embedding_provider(provider_factory=lambda **kwargs: model) as result:
        assert result is model


def test_bad_context_size_does_not_construct_provider():
    factory = Mock()
    with pytest.raises(ValueError, match='context_size'):
        with application.open_embedding_provider(context_size=0, provider_factory=factory):
            pass
    factory.assert_not_called()


def test_ingestion_application_releases_context_after_command_failure(monkeypatch):
    backend = Mock()
    context = ApplicationContext(documents=Mock(), storage=Mock(), database=backend)
    monkeypatch.setattr(application, 'create_context', lambda: context)
    model = Mock()
    monkeypatch.setattr(application, 'get_embedding_provider', lambda **kwargs: model)
    command = Mock()
    command.execute.side_effect = ValueError('invalid input')
    monkeypatch.setattr(application, 'IngestDocument', lambda **kwargs: command)
    with pytest.raises(ValueError, match='invalid input'):
        application.ingest_file(path='unused', chunk_size=1, stride=1)
    backend.close.assert_called_once()
    model.close.assert_called_once()
