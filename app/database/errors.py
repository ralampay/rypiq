"""Translate PostgreSQL adapter failures without exposing driver diagnostics."""
from contextlib import contextmanager
from collections.abc import Iterator
from uuid import UUID

import sqlalchemy as sa

from app.exceptions import (
    DatabaseConnectionError, DatabaseError, DatabaseIntegrityError,
    DuplicateDocumentNameError, RyPiqError,
)


_CLEANUP_NOTE = 'Database rollback or connection cleanup failed; resources may remain open.'


def _translate(error: sa.exc.SQLAlchemyError, *, application_id: UUID | None,
               name: str | None) -> RyPiqError:
    original = getattr(error, 'orig', None)
    state = getattr(original, 'sqlstate', None)
    constraint = getattr(getattr(original, 'diag', None), 'constraint_name', None)
    # Classify using stable PostgreSQL codes, never driver messages or SQL values.
    if state in ('42703', '42P01'):
        missing = 'column' if state == '42703' else 'table'
        return DatabaseError(
            f'Database schema is out of date or incompatible: a required {missing} is missing. '
            'Run rypiq --doctor to identify missing schema objects, then rypiq db:migrate '
            'to apply supported upgrades. Incompatible schemas require a deliberate schema change.')
    if state == '42501':
        return DatabaseError(
            'Database permission denied; check privileges for the configured database user '
            'on the selected database, schema, and tables.')
    if isinstance(error, sa.exc.IntegrityError):
        if state == '23505' and constraint == 'uq_documents_application_name':
            return DuplicateDocumentNameError(application_id=application_id, name=name)
        return DatabaseIntegrityError('Database constraint rejected the operation')
    if (state is not None and (state.startswith('08') or state.startswith('28') or state == '3D000')
            or getattr(error, 'connection_invalidated', False)
            or isinstance(error, (sa.exc.OperationalError, sa.exc.InterfaceError)) and state is None):
        return DatabaseConnectionError(
            'Cannot connect to database; check connectivity and credentials. '
            'If it does not exist, run db:create first.')
    return DatabaseError('Database operation failed; check connectivity, permissions, '
                         'pgvector availability and schema configuration')


@contextmanager
def database_errors(*, application_id: UUID | None = None,
                    name: str | None = None) -> Iterator[None]:
    try:
        yield
    except sa.exc.SQLAlchemyError as error:
        translated = _translate(error, application_id=application_id, name=name)
        # Only copy our own safe note, never arbitrary driver diagnostics.
        if _CLEANUP_NOTE in getattr(error, '__notes__', ()):
            translated.add_note(_CLEANUP_NOTE)
        raise translated from None


@contextmanager
def database_connection(engine: sa.Engine, *, write: bool = False,
                        application_id: UUID | None = None,
                        name: str | None = None) -> Iterator[sa.Connection]:
    """Cover acquire/execute/commit, preserving the primary failure on cleanup."""
    with database_errors(application_id=application_id, name=name):
        failure = None
        try:
            with (engine.begin() if write else engine.connect()) as connection:
                try:
                    yield connection
                except BaseException as exc:
                    failure = exc
                    raise
        except BaseException as exc:
            if failure is not None and exc is not failure:
                failure.add_note(_CLEANUP_NOTE)
                raise failure from None
            raise
