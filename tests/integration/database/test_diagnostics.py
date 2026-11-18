from app.database.diagnostics import check_database
from app.database.postgresql_connection import create_database


def test_doctor_inspects_initialized_database_without_schema_changes(backend):
    import sqlalchemy as sa

    backend.create_database()
    before = sa.inspect(backend.engine).get_table_names(schema='public')
    results = check_database(settings=backend.settings)
    assert [result.status for result in results] == ['COMPLETE', 'COMPLETE', 'COMPLETE']
    assert sa.inspect(backend.engine).get_table_names(schema='public') == before
    assert backend.applications().list() == []
    assert backend.documents().list() == []


def test_doctor_reports_absent_database_without_creating_it(backend):
    results = check_database(settings=backend.settings)
    assert len(results) == 1
    assert results[0].status == 'INVALID'
    assert 'db:create' in results[0].detail


def test_doctor_reports_uninitialized_database_without_migrating(backend):
    import sqlalchemy as sa

    create_database(backend.maintenance_url, backend.url.database)
    results = check_database(settings=backend.settings)
    assert results[0].status == 'COMPLETE'
    assert results[-1].status == 'REQUIRED'
    assert 'db:migrate' in results[-1].detail
    assert sa.inspect(backend.engine).get_table_names(schema='public') == []
