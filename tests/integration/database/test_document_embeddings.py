"""Real PostgreSQL coverage for aggregate persistence and the chunk-table upgrade."""
from dataclasses import asdict, replace
from uuid import uuid4

import pytest
import sqlalchemy as sa

from app.database.legacy_schema import build_metadata
from app.database.postgresql_connection import create_database
from app.models.application import Application
from app.models.document import Document, utc_now
from app.models.embedding import Embedding


def test_aggregate_atomicity_cascade_and_chunk_metadata_search(backend):
    backend.create_database()
    application = backend.applications().create(Application(name='a', embedding_model='m', embedding_size=2))
    parent = Document(application_id=application.id, storage_key='keep')
    first = Embedding(document_id=parent.id, content='first', embedding=[1., 0.],
                      embedding_model='m', embedding_size=2, metadata={'tag': {'language': 'en'}})
    second = replace(first, id=uuid4(), content='second', chunk_index=1, embedding=[0., 1.], metadata={'tag': {'language': 'fr'}})
    # Duplicate chunk IDs fail after the parent insert; the transaction must undo both.
    with pytest.raises(sa.exc.IntegrityError):
        backend.documents().create_with_embeddings(parent, [first, replace(second, id=first.id)])
    assert backend.documents().get(parent.id) is None
    assert backend.embeddings().get(first.id) is None
    backend.documents().create_with_embeddings(parent, [second, first])
    details = backend.documents().details(parent.id)
    assert [item.content for item in details.embeddings] == ['first', 'second']
    matches = backend.embeddings().search_by_vector([1., 0.], application_id=application.id,
                                                    metadata_filters={'tag': {'language': 'fr'}})
    assert len(matches) == 1 and matches[0].embedding.id == second.id
    assert matches[0].document.id == parent.id
    assert matches[0].distance == pytest.approx(1.)
    backend.embeddings().delete(first.id)
    assert backend.documents().get(parent.id) is not None
    assert backend.embeddings().get(second.id) is not None
    backend.embeddings().delete(second.id)
    assert backend.documents().details(parent.id).embeddings == []
    backend.embeddings().create(first)
    backend.documents().delete(parent.id)
    assert backend.embeddings().get(first.id) is None


def seed_current_legacy(backend):
    create_database(backend.maintenance_url, backend.url.database)
    metadata = build_metadata()
    app = Application(name='old', embedding_model='m', embedding_size=2)
    source_id = uuid4()
    now = utc_now()
    common = dict(application_id=app.id, content='chunk', embedding=[1., 0.],
                  embedding_model='m', embedding_size=2, provision_type='local', model_name='m',
                  storage_key='keep', filename='original.txt', content_type='text/plain', source='original.txt',
                  source_id=source_id, metadata={'storage_checksum': 'same', 'start': 0, 'custom': {'keep': True}},
                  created_at=now, updated_at=now)
    rows = [dict(common, id=uuid4(), chunk_index=i) for i in range(5)]
    rows[2]['storage_key'] = 'different-file'
    rows[3]['source_id'] = rows[4]['source_id'] = None
    with backend.engine.begin() as connection:
        connection.exec_driver_sql('CREATE EXTENSION vector')
        metadata.create_all(connection)
        connection.execute(metadata.tables['public.applications'].insert().values(**asdict(app)))
        connection.execute(metadata.tables['public.documents'].insert(), rows)
    return rows


def test_current_legacy_upgrade_preserves_ids_and_splits_ambiguous_groups(backend):
    rows = seed_current_legacy(backend)
    assert backend.migrate() == {'tables_created': ['embeddings']}
    assert backend.migrate() == {'tables_created': []}
    parents = backend.documents().list()
    assert len(parents) == 4
    chunks = [backend.embeddings().get(row['id']) for row in rows]
    assert chunks[0].document_id == chunks[1].document_id
    assert len({item.document_id for item in chunks[2:]}) == 3
    for original, chunk in zip(rows, chunks):
        assert chunk.metadata == original['metadata']
        assert chunk.content == original['content']
        assert chunk.embedding == original['embedding']
        assert chunk.created_at == original['created_at']
        parent = backend.documents().get(chunk.document_id)
        assert parent.storage_key == original['storage_key']
        assert parent.metadata['source_id'] == (str(original['source_id']) if original['source_id'] else None)
        assert parent.metadata['storage_checksum'] == 'same'


def test_current_legacy_split_failure_restores_original_table(backend):
    rows = seed_current_legacy(backend)
    sa.Table('z_broken', backend.metadata, sa.Column('id', sa.Integer, server_default=sa.text('missing_function()')))
    with pytest.raises(sa.exc.ProgrammingError):
        backend.migrate()
    assert set(sa.inspect(backend.engine).get_table_names()) == {'applications', 'documents'}
    with backend.engine.connect() as connection:
        assert connection.exec_driver_sql('SELECT count(*) FROM documents').scalar_one() == len(rows)


def test_invalid_metadata_filter_fails_before_database_connection(backend):
    with pytest.raises(ValueError, match='JSON object'):
        backend.embeddings().search_by_vector([1., 0.], application_id=uuid4(), metadata_filters=[])


def test_chunk_service_scope_model_snapshots_and_immutable_parent(backend):
    from app.services.embedding import EmbeddingService

    backend.create_database()
    first = backend.applications().create(Application(name='first', embedding_model='m', embedding_size=2))
    second = backend.applications().create(Application(name='second', embedding_model='n', embedding_size=2))
    parent = backend.documents().create(Document(application_id=first.id))
    sibling_parent = backend.documents().create(Document(application_id=first.id))
    outsider = backend.documents().create(Document(application_id=second.id))
    service = EmbeddingService(application_id=first.id, repository=backend.embeddings())
    chunk = Embedding(document_id=parent.id, content='chunk', embedding=[1., 0.],
                      embedding_model='m', embedding_size=2)
    service.create(chunk)
    other = EmbeddingService(application_id=second.id, repository=backend.embeddings())
    with pytest.raises(KeyError):
        other.get(chunk.id)
    assert other.delete(chunk.id) is False
    assert other.list(document_id=parent.id) == []
    with pytest.raises(KeyError):
        other.update(chunk)
    with pytest.raises(KeyError):
        service.create(replace(chunk, id=uuid4(), document_id=outsider.id))
    with pytest.raises(ValueError, match='ownership'):
        service.update(replace(chunk, document_id=sibling_parent.id))
    with pytest.raises(ValueError, match='match'):
        service.update(replace(chunk, embedding_model='wrong'))
    with pytest.raises(KeyError):
        service.update(replace(chunk, id=uuid4()))
    updated = service.update(replace(chunk, content='updated', metadata={'tag': 'value'}))
    assert updated.updated_at >= chunk.updated_at
    assert service.get(chunk.id).content == 'updated'
    assert service.list(document_id=parent.id)[0].metadata == {'tag': 'value'}


def test_aggregate_snapshot_mismatch_does_not_persist_parent(backend):
    backend.create_database()
    application = backend.applications().create(Application(name='a', embedding_model='m', embedding_size=2))
    parent = Document(application_id=application.id)
    wrong = Embedding(document_id=parent.id, content='text', embedding=[1., 0.],
                      embedding_model='wrong', embedding_size=2)
    with pytest.raises(ValueError, match='match'):
        backend.documents().create_with_embeddings(parent, [wrong])
    assert backend.documents().get(parent.id) is None


def test_upgrade_never_merges_different_applications(backend):
    rows = seed_current_legacy(backend)
    metadata = build_metadata()
    another = Application(name='other', embedding_model='m', embedding_size=2)
    duplicate = dict(rows[0], id=uuid4(), application_id=another.id)
    with backend.engine.begin() as connection:
        connection.execute(metadata.tables['public.applications'].insert().values(**asdict(another)))
        connection.execute(metadata.tables['public.documents'].insert().values(**duplicate))
    backend.migrate()
    original = backend.embeddings().get(rows[0]['id'])
    migrated = backend.embeddings().get(duplicate['id'])
    assert original.document_id != migrated.document_id
    assert backend.documents().get(migrated.document_id).application_id == another.id
