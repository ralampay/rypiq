from unittest.mock import MagicMock, Mock

import pytest
import sqlalchemy as sa

from app.database import diagnostics
from app.database.schema import build_metadata


@pytest.fixture
def connection(monkeypatch):
    engine = MagicMock()
    connected = engine.connect.return_value.__enter__.return_value
    connected.execute.return_value.scalar_one.return_value = True
    inspector = Mock()
    metadata = build_metadata()
    inspector.get_table_names.return_value = ['applications', 'documents', 'embeddings', 'document_processing_jobs']
    inspector.get_columns.side_effect = lambda name, **kwargs: [
        {'name': column.name} for column in metadata.tables['public.' + name].columns]
    monkeypatch.setattr(diagnostics.sa, 'inspect', Mock(return_value=inspector))
    factory = Mock(return_value=engine)
    monkeypatch.setattr(diagnostics.sa, 'create_engine', factory)
    return engine, connected, inspector, factory


def check():
    return diagnostics.check_database(settings={'adapter': 'postgresql', 'host': 'localhost',
        'database': 'example', 'user': 'example', 'password': 'private-password', 'dimensions': 768})


def test_database_uses_read_only_timeouts_and_releases_resources(connection):
    engine, connected, _, factory = connection
    results = check()
    assert [item.name for item in results] == ['Database connection', 'pgvector extension', 'Database tables']
    assert all(item.status == 'COMPLETE' for item in results)
    options = factory.call_args.kwargs['connect_args']
    assert options['connect_timeout'] == 5
    assert 'default_transaction_read_only=on' in options['options']
    assert 'statement_timeout=5000' in options['options']
    assert all(str(call.args[0]).lstrip().startswith('SELECT') for call in connected.execute.call_args_list)
    engine.connect.return_value.__exit__.assert_called_once()
    engine.dispose.assert_called_once()


@pytest.mark.parametrize('state, expected', [('3D000', 'db:create'), ('28P01', 'credentials'), (None, 'connect')])
def test_connection_failures_are_sanitized(connection, state, expected):
    engine, _, _, _ = connection
    driver = Exception('private-password postgresql://secret@host/database')
    driver.sqlstate = state
    engine.connect.side_effect = sa.exc.OperationalError('private SQL', {}, driver)
    results = check()
    assert len(results) == 1 and results[0].status == 'INVALID'
    assert expected in results[0].detail
    assert 'private-password' not in repr(results) and 'postgresql://' not in repr(results)
    engine.dispose.assert_called_once()


def test_missing_extension_and_tables_are_reported_independently(connection):
    _, connected, inspector, _ = connection
    connected.execute.return_value.scalar_one.return_value = False
    inspector.get_table_names.return_value = []
    results = check()
    assert [item.status for item in results] == ['COMPLETE', 'REQUIRED', 'REQUIRED']
    assert 'pgvector' in results[1].detail
    assert 'applications' in results[2].detail and 'documents' in results[2].detail


def test_missing_columns_are_reported(connection):
    _, _, inspector, _ = connection
    inspector.get_columns.side_effect = lambda *args, **kwargs: [{'name': 'id'}]
    results = check()
    assert results[-1].status == 'INVALID'
    assert 'application_id' in results[-1].detail and 'embedding' in results[-1].detail


def test_query_failure_is_sanitized_and_connection_closed(connection):
    engine, connected, _, _ = connection
    connected.execute.side_effect = sa.exc.OperationalError('private SQL', {}, Exception('private-password'))
    results = check()
    assert results[-1].status == 'INVALID'
    assert 'private-password' not in repr(results)
    engine.connect.return_value.__exit__.assert_called_once()
    engine.dispose.assert_called_once()
