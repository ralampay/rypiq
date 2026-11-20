from unittest.mock import Mock

import pytest

from app import db
from app.database.repositories.postgresql.document import PostgreSQLDocumentRepository
from app.database.schema import build_metadata


def test_legacy_repository_factory_returns_owned_repository(monkeypatch):
    backend = Mock()
    monkeypatch.setattr(db, 'get_database', lambda config=None: backend)
    assert db.get_document_repository() is backend.documents.return_value
    backend.close.assert_not_called()


def test_legacy_repository_factory_closes_after_setup_failure(monkeypatch):
    backend = Mock()
    backend.documents.side_effect = ValueError('cannot construct repository')
    monkeypatch.setattr(db, 'get_database', lambda config=None: backend)
    with pytest.raises(ValueError):
        db.get_document_repository()
    backend.close.assert_called_once()


def test_owned_repository_can_release_its_engine():
    engine = Mock()
    repository = PostgreSQLDocumentRepository(engine, build_metadata(3).tables['public.documents'])
    repository.close()
    engine.dispose.assert_called_once()


@pytest.mark.parametrize('limit,offset', [(0, 0), (True, 0), (1, -1), (1, False)])
def test_pagination_rejects_invalid_inputs_before_io(limit, offset):
    engine = Mock()
    repository = PostgreSQLDocumentRepository(engine, build_metadata(3).tables['public.documents'])
    with pytest.raises(ValueError):
        repository.list(limit, offset)
    engine.connect.assert_not_called()
