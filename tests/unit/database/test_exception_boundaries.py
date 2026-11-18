from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import sqlalchemy as sa

from app.database.backends.postgresql import PostgreSQLBackend
from app.database.repositories.postgresql.application import PostgreSQLApplicationRepository
from app.database.schema import build_metadata


@pytest.mark.parametrize('failure, expected', [
    (sa.exc.OperationalError('secret SQL', {}, Exception('secret password')), 'DatabaseConnectionError'),
    (sa.exc.IntegrityError('secret SQL', {}, Exception('secret value')), 'DatabaseIntegrityError'),
])
def test_repository_translates_driver_failures(failure, expected):
    engine = Mock()
    engine.connect.side_effect = failure
    repository = PostgreSQLApplicationRepository(engine, build_metadata(3).tables['public.applications'])
    with pytest.raises(Exception) as caught:
        repository.get('unused')
    assert type(caught.value).__name__ == expected
    assert 'secret' not in str(caught.value)
    assert caught.value.__suppress_context__


def test_reset_exposes_completed_step(monkeypatch):
    backend = PostgreSQLBackend(dict(adapter='postgresql', host='localhost', user='tester', database='example'))
    monkeypatch.setattr(backend, 'drop_database', lambda: {'dropped': True})
    monkeypatch.setattr(backend, 'create_database', Mock(side_effect=OSError('secret')))
    try:
        with pytest.raises(RuntimeError) as caught:
            backend.reset()
        assert type(caught.value).__name__ == 'DatabaseResetError'
        assert caught.value.completed_steps == ('drop',)
        assert caught.value.operation == 'db:reset'
    finally:
        backend.close()


@pytest.mark.parametrize('stage', ['acquire', 'execute', 'commit'])
def test_transaction_failures_are_translated_at_every_stage(stage):
    from unittest.mock import MagicMock
    from app.exceptions import DatabaseError
    from app.database.errors import database_connection
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    failure = sa.exc.ProgrammingError('secret SQL', {}, Exception('secret detail'))
    if stage == 'acquire':
        engine.begin.return_value.__enter__.side_effect = failure
    elif stage == 'execute':
        connection.execute.side_effect = failure
    else:
        engine.begin.return_value.__exit__.side_effect = failure
    with pytest.raises(DatabaseError) as caught:
        with database_connection(engine, write=True) as active:
            active.execute('statement')
    assert 'secret' not in str(caught.value)
    assert caught.value.__suppress_context__


def test_cleanup_preserves_primary_domain_failure():
    from unittest.mock import MagicMock
    from app.database.errors import database_connection
    from app.exceptions import ValidationError
    engine = MagicMock()
    engine.begin.return_value.__exit__.side_effect = OSError('sensitive cleanup')
    primary = ValidationError('bad input')
    with pytest.raises(ValidationError) as caught:
        with database_connection(engine, write=True):
            raise primary
    assert caught.value is primary
    assert 'rollback' in primary.__notes__[0]
    assert 'sensitive' not in primary.__notes__[0]


def test_constraint_race_has_same_exception_and_context_as_preflight():
    from app.database.errors import database_errors
    from app.exceptions import DuplicateDocumentNameError
    from app.operations.documents.name import prepare_document_name
    from uuid import uuid4
    identity = uuid4()
    repository = Mock()
    repository.name_exists.return_value = True
    with pytest.raises(DuplicateDocumentNameError) as before:
        prepare_document_name(name=' report ', application_id=identity, repository=repository)
    driver = Exception('sensitive SQL value')
    driver.sqlstate = '23505'
    driver.diag = SimpleNamespace(constraint_name='uq_documents_application_name')
    with pytest.raises(DuplicateDocumentNameError) as race:
        with database_errors(application_id=identity, name='report'):
            raise sa.exc.IntegrityError('sensitive SQL', {}, driver)
    for error in (before.value, race.value):
        assert error.application_id == identity
        assert error.name == 'report'
        assert 'sensitive' not in str(error)
    assert type(before.value) is type(race.value)


def test_cleanup_note_survives_driver_error_translation():
    from unittest.mock import MagicMock
    from app.database.errors import database_connection
    from app.exceptions import DatabaseError
    engine = MagicMock()
    engine.begin.return_value.__exit__.side_effect = OSError('secret cleanup')
    with pytest.raises(DatabaseError) as caught:
        with database_connection(engine, write=True):
            raise sa.exc.ProgrammingError('secret statement', {}, Exception('secret primary'))
    assert 'rollback' in caught.value.__notes__[0]
    assert 'secret' not in str(caught.value)


@pytest.mark.parametrize('created', [True, False])
def test_schema_failure_reports_only_completed_database_creation(monkeypatch, created):
    from app.database.backends import postgresql
    from app.exceptions import DatabaseInitializationError, SchemaCompatibilityError
    backend = PostgreSQLBackend(dict(adapter='postgresql', host='localhost', user='tester', database='example'))
    primary = SchemaCompatibilityError('incompatible schema')
    monkeypatch.setattr(postgresql, 'create_database', lambda *args: {'created': created})
    monkeypatch.setattr(backend, 'migrate', Mock(side_effect=primary))
    try:
        with pytest.raises((DatabaseInitializationError, SchemaCompatibilityError)) as caught:
            backend.create_database()
        if created:
            assert isinstance(caught.value, DatabaseInitializationError)
            assert caught.value.completed_steps == ('create_database',)
            assert caught.value.operation == 'db:create'
        else:
            assert caught.value is primary
    finally:
        backend.close()


@pytest.mark.parametrize('state, expected', [
    ('08006', 'DatabaseConnectionError'), ('28P01', 'DatabaseConnectionError'),
    ('3D000', 'DatabaseConnectionError'), ('55P03', 'DatabaseError'),
])
def test_database_error_classification_uses_sqlstate_without_exposing_driver(state, expected):
    from app.database.errors import database_errors
    from app.exceptions import RyPiqError
    driver = Exception('secret connection string')
    driver.sqlstate = state
    with pytest.raises(RyPiqError) as caught:
        with database_errors():
            raise sa.exc.OperationalError('secret statement', {}, driver)
    assert type(caught.value).__name__ == expected
    assert caught.value.__suppress_context__
    assert 'secret' not in str(caught.value)


def test_application_delete_with_documents_raises_conflict_before_delete():
    from unittest.mock import MagicMock
    from uuid import uuid4
    from app.exceptions import ApplicationInUseError
    engine = MagicMock()
    connection = engine.begin.return_value.__enter__.return_value
    identity = uuid4()
    connection.execute.return_value.first.return_value = (identity,)
    connection.execute.return_value.scalar.return_value = True
    repository = PostgreSQLApplicationRepository(engine, build_metadata(3).tables['public.applications'])
    with pytest.raises(ApplicationInUseError):
        repository.delete(identity)
    assert connection.execute.call_count == 2  # Parent lookup and document check only.
