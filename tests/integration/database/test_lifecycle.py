from app.exceptions import DatabaseError
import pytest
import sqlalchemy as sa


def test_create_installs_extension_and_fixed_schema(backend):
    result = backend.create_database()
    assert result['created'] is True
    with backend.engine.connect() as connection:
        assert connection.execute(sa.text("SELECT extname FROM pg_extension WHERE extname='vector'")).scalar() == 'vector'
    inspector = sa.inspect(backend.engine)
    columns = {c['name']: c for c in inspector.get_columns('embeddings')}
    assert str(columns['embedding']['type']) == 'VECTOR'
    assert 'model_name' in columns
    assert 'embedding_model' in columns
    assert set(inspector.get_table_names()) == {'applications', 'document_processing_jobs', 'documents', 'embeddings'}
    assert backend.create_database()['created'] is False
    assert backend.migrate() == {'tables_created': []}


def test_migrate_missing_database_does_not_create_it(backend):
    with pytest.raises(ValueError, match='db:create'):
        backend.migrate()


def test_migrate_recreates_missing_table_without_dropping_unrelated_data(backend):
    backend.create_database()
    with backend.engine.begin() as connection:
        connection.exec_driver_sql('DROP TABLE embeddings')
        connection.exec_driver_sql('CREATE TABLE unrelated (value text)')
        connection.exec_driver_sql("INSERT INTO unrelated VALUES ('keep')")
    assert backend.migrate() == {'tables_created': ['embeddings']}
    with backend.engine.connect() as connection:
        assert connection.exec_driver_sql('SELECT value FROM unrelated').scalar() == 'keep'


@pytest.mark.parametrize('change', [
    'ALTER TABLE documents DROP COLUMN filename',
    'ALTER TABLE documents ADD COLUMN unexpected text',
    'ALTER TABLE documents ALTER COLUMN filename TYPE varchar(50)',
    'ALTER TABLE documents ALTER COLUMN filename DROP NOT NULL',
    'ALTER TABLE documents DROP CONSTRAINT pk_documents CASCADE',
    'ALTER TABLE embeddings ALTER COLUMN embedding TYPE vector(4)',
])
def test_incompatible_existing_schema_is_not_silently_changed(backend, change):
    backend.create_database()
    with backend.engine.begin() as connection:
        connection.exec_driver_sql(change)
    with pytest.raises(ValueError, match='Incompatible'):
        backend.migrate()


def test_drop_reset_preserve_other_database_and_files(backend, tmp_path):
    sentinel = tmp_path / 'object.txt'
    sentinel.write_text('keep')
    backend.create_database()
    with backend.engine.begin() as connection:
        connection.exec_driver_sql('CREATE TABLE unrelated (id integer)')
    assert backend.reset()['created'] is True
    assert set(sa.inspect(backend.engine).get_table_names()) == {'applications', 'document_processing_jobs', 'documents', 'embeddings'}
    assert backend.drop_database() == {'dropped': True}
    assert backend.drop_database() == {'dropped': False}
    assert sentinel.read_text() == 'keep'
    admin = sa.create_engine(backend.url.set(database=backend.settings['maintenance_database']))
    try:
        with admin.connect() as connection:
            assert connection.exec_driver_sql('SELECT 1').scalar() == 1
    finally:
        admin.dispose()


def test_drop_does_not_terminate_external_connections(backend):
    backend.create_database()
    external_engine = sa.create_engine(backend.url)
    try:
        with external_engine.connect() as external:
            with pytest.raises(DatabaseError) as failure:
                backend.drop_database()
            assert failure.value.__suppress_context__
            assert external.exec_driver_sql('SELECT 1').scalar() == 1
    finally:
        external_engine.dispose()
    assert backend.drop_database() == {'dropped': True}


def test_schema_initialization_is_atomic(backend):
    from app.database.postgresql_connection import create_database
    create_database(backend.maintenance_url, backend.url.database)
    sa.Table('z_broken', backend.metadata,
             sa.Column('id', sa.Integer, server_default=sa.text('missing_function()')))
    with pytest.raises(DatabaseError):
        backend.migrate()
    assert sa.inspect(backend.engine).get_table_names() == []
    with backend.engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT extname FROM pg_extension WHERE extname='vector'").scalar() is None


def test_missing_tables_not_created_when_existing_table_is_incompatible(backend):
    backend.create_database()
    sa.Table('extra', backend.metadata, sa.Column('id', sa.Integer, primary_key=True))
    with backend.engine.begin() as connection:
        connection.exec_driver_sql('ALTER TABLE documents DROP COLUMN filename')
    with pytest.raises(ValueError, match='Incompatible'):
        backend.migrate()
    assert 'extra' not in sa.inspect(backend.engine).get_table_names()
