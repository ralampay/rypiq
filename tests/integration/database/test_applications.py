from dataclasses import asdict
from uuid import uuid4

import pytest
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from app.models.application import Application
from app.models.document import Document
from app.database.postgresql_connection import create_database


def owned(application, vector):
    return Document(content='content', embedding=vector, application_id=application.id,
                    embedding_model=application.embedding_model, embedding_size=application.embedding_size,
                    provision_type=application.provision_type)


def test_applications_and_mixed_size_search(backend):
    backend.create_database()
    applications, documents = backend.applications(), backend.documents()
    first = applications.create(Application(name='first', embedding_model='a', embedding_size=2,
                                            data={'nested': [1, {'two': True}]}))
    second = applications.create(Application(name='second', embedding_model='b', embedding_size=3))
    assert asdict(applications.get(first.id)) == asdict(first)
    assert len(applications.list()) == 2
    a, b = owned(first, [1., 0.]), owned(second, [1., 0., 0.])
    documents.create(a)
    documents.create(b)
    assert [r.document.id for r in documents.search_by_vector([1., 0.], application_id=first.id)] == [a.id]
    assert [r.document.id for r in documents.search_by_vector([1., 0., 0.], application_id=second.id)] == [b.id]
    assert documents.search_by_vector([1., 0.], application_id=first.id,
                                      filters={'application_id': second.id}) == []
    with pytest.raises(ValueError, match='exactly 2'):
        documents.search_by_vector([1., 0., 0.], application_id=first.id)
    a.content = 'changed'
    documents.update(a)
    assert documents.get(a.id).embedding_model == 'a'
    a.embedding_model = 'wrong'
    with pytest.raises(ValueError, match='match'):
        documents.update(a)
    missing = owned(first, [1., 0.])
    missing.application_id = uuid4()
    with pytest.raises(KeyError):
        documents.create(missing)
    table = backend.metadata.tables['public.applications']
    with pytest.raises(sa.exc.IntegrityError):
        with backend.engine.begin() as connection:
            connection.execute(table.delete().where(table.c.id == first.id))
    second.api_key = first.api_key
    second.id = uuid4()
    with pytest.raises(sa.exc.IntegrityError):
        applications.create(second)


def create_legacy(backend):
    create_database(backend.maintenance_url, backend.url.database)
    current = backend.metadata.tables['public.documents']
    metadata = sa.MetaData(schema='public')
    table = sa.Table('documents', metadata, *(
        sa.Column(c.name, Vector(3) if c.name == 'embedding' else c.type,
                  nullable=c.nullable, primary_key=c.primary_key)
        for c in current.c if c.name not in ('application_id', 'embedding_model', 'embedding_size', 'provision_type')))
    originals = []
    with backend.engine.begin() as connection:
        connection.exec_driver_sql('CREATE EXTENSION vector')
        metadata.create_all(connection)
        for model in ('a', 'a', 'b', ''):
            value = asdict(Document(content=model, model_name=model, embedding=[1., 0., 0.],
                                    metadata={'keep': [1, 2]}, storage_key='keep.txt'))
            value = {key: value for key, value in value.items() if key in table.c}
            connection.execute(table.insert().values(**value))
            originals.append(value)
    return originals


def test_legacy_upgrade_groups_models_and_preserves_values(backend):
    originals = create_legacy(backend)
    assert backend.migrate() == {'tables_created': ['applications']}
    assert backend.migrate() == {'tables_created': []}
    applications = backend.applications().list()
    assert len(applications) == 3
    assert len({a.api_key for a in applications}) == 3
    for original in originals:
        loaded = asdict(backend.documents().get(original['id']))
        assert {key: loaded[key] for key in original} == original
        app = backend.applications().get(loaded['application_id'])
        assert app.embedding_model == original['model_name'] == loaded['embedding_model']
        assert app.embedding_size == loaded['embedding_size'] == 3
        assert loaded['provision_type'] == 'local'
    assert next(a for a in applications if not a.embedding_model).data['requires_model_configuration']


def test_failed_upgrade_rolls_back_all_changes(backend):
    create_legacy(backend)
    sa.Table('z_broken', backend.metadata,
             sa.Column('id', sa.Integer, server_default=sa.text('missing_function()')))
    with pytest.raises(sa.exc.ProgrammingError):
        backend.migrate()
    inspector = sa.inspect(backend.engine)
    assert inspector.get_table_names() == ['documents']
    assert 'application_id' not in {c['name'] for c in inspector.get_columns('documents')}
    with backend.engine.connect() as connection:
        assert connection.exec_driver_sql('SELECT count(*) FROM documents').scalar() == 4


@pytest.mark.parametrize('change', [
    'ALTER TABLE documents DROP CONSTRAINT ck_documents_vector_size',
    'ALTER TABLE documents DROP CONSTRAINT documents_application_id_fkey',
    'ALTER TABLE applications DROP CONSTRAINT applications_api_key_key',
    'ALTER TABLE applications DROP CONSTRAINT ck_applications_provider',
])
def test_constraint_drift_is_rejected(backend, change):
    backend.create_database()
    with backend.engine.begin() as connection:
        connection.exec_driver_sql(change)
    with pytest.raises(ValueError, match='Incompatible'):
        backend.migrate()


def test_reset_after_upgrade_retains_constraints(backend):
    create_legacy(backend)
    backend.migrate()
    backend.reset()
    assert backend.migrate() == {'tables_created': []}
    inspector = sa.inspect(backend.engine)
    assert inspector.get_foreign_keys('documents')
    assert len(inspector.get_check_constraints('documents')) == 3


def test_unrecognized_legacy_schema_is_not_upgraded(backend):
    create_legacy(backend)
    with backend.engine.begin() as connection:
        connection.exec_driver_sql('ALTER TABLE documents ADD COLUMN unexpected text')
    with pytest.raises(ValueError, match='Incompatible'):
        backend.migrate()
    assert sa.inspect(backend.engine).get_table_names() == ['documents']
