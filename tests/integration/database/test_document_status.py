from contextlib import nullcontext
from dataclasses import asdict, replace
from unittest.mock import Mock
from uuid import uuid4

import pytest
import sqlalchemy as sa

from app.database.postgresql_connection import create_database
from app.database.pre_status_schema import build_metadata
from app.exceptions import ConflictError, DatabaseIntegrityError, SchemaCompatibilityError
from app.models.application import Application
from app.models.document import Document
from app.models.embedding import Embedding
from app.models.processing import SNAPSHOT_KEY
from app.operations.documents.create import CreateDocument
from app.operations.documents.process import ProcessDocument
from app.services.document import DocumentService
from app.storage.providers.local import LocalStorage


@pytest.fixture
def processing(backend, tmp_path):
    backend.create_database()
    application = backend.applications().create(Application(name='test', embedding_model='old', embedding_size=2,
        data={'chunking': {'chunk_size': 5, 'stride': 5}}))
    storage = LocalStorage(tmp_path / 'objects')
    source = tmp_path / 'source.txt'
    source.write_text('abcdefghij')
    repository = backend.documents()
    created = CreateDocument(name='Report', path=str(source), application=application,
        repository=repository, storage=storage).execute()
    provider = Mock(model_name='old')
    provider.embed.side_effect = lambda texts: [[1., 0.] for text in texts]
    def command(scope=None):
        return ProcessDocument(document_id=created.document.id, repository=repository,
            applications=backend.applications(), storage=storage,
            provider_scope=scope or (lambda application: nullcontext(provider)))
    return application, repository, storage, source, created.document, provider, command


def test_pending_processing_done_and_retry_from_stored_original(processing, backend):
    application, repository, storage, source, document, provider, command = processing
    assert repository.get(document.id).status == 'pending'
    source.unlink()
    def embed(texts):
        assert repository.get(document.id).status == 'processing'
        return [[1., 0.] for text in texts]
    provider.embed.side_effect = embed
    result = command().execute()
    assert result.document.status == 'done'
    assert len(result.embeddings) == 2
    before = [item.id for item in result.embeddings]
    provider.embed.side_effect = RuntimeError('provider failed')
    with pytest.raises(RuntimeError, match='provider failed'):
        command().execute()
    assert repository.get(document.id).status == 'error'
    assert [item.id for item in repository.details(document.id).embeddings] == before
    assert len(backend.embeddings().search_by_vector([1., 0.], application_id=application.id)) == 2
    assert storage.exists(document.storage_key)
    provider.embed.side_effect = embed
    retry = command().execute()
    assert retry.document.id == document.id
    assert retry.document.created_at == document.created_at
    assert not set(before) & {item.id for item in retry.embeddings}
    assert len(storage.list()) == 1
    provider.close.assert_not_called()


def test_filters_are_scoped_and_validate_status(processing, backend):
    application, repository, _, _, document, _, command = processing
    other = backend.applications().create(Application(name='other', embedding_model='old'))
    repository.create(Document(name='outside', application_id=other.id))
    service = DocumentService(application_id=application.id, repository=repository, applications=backend.applications())
    assert [item.id for item in service.list(status='pending')] == [document.id]
    assert repository.list_summaries(application_id=application.id, status='pending')[0].status == 'pending'
    command().execute()
    assert service.list(status='pending') == []
    assert service.list(status='done')[0].id == document.id
    with pytest.raises(ValueError, match='status'):
        service.list(status='unknown')
    with backend.engine.begin() as connection:
        with pytest.raises(sa.exc.IntegrityError):
            connection.execute(backend.metadata.tables['public.documents'].update().values(status='bad'))


def test_replace_is_atomic_and_config_changes_require_reprocessing(processing, backend):
    application, repository, _, _, document, provider, command = processing
    original = command().execute()
    chunks = original.embeddings
    with pytest.raises(DatabaseIntegrityError):
        repository.replace_embeddings(original.document, [chunks[0], chunks[0]], application=application)
    assert repository.details(document.id) == original
    changed = backend.applications().update(replace(application, embedding_model='new', embedding_size=3))
    assert backend.embeddings().search_by_vector([1., 0., 0.], application_id=application.id) == []
    with pytest.raises(ConflictError, match='changed'):
        repository.replace_embeddings(original.document, chunks, application=application)
    assert repository.details(document.id) == original
    provider.model_name = 'new'
    provider.embed.side_effect = lambda texts: [[1., 0., 0.] for text in texts]
    result = command().execute()
    assert all(item.embedding_size == 3 and item.embedding_model == 'new' for item in result.embeddings)
    assert len(backend.embeddings().search_by_vector([1., 0., 0.], application_id=application.id)) == 2
    # Same-dimensional models must not be compared either.
    backend.applications().update(replace(changed, embedding_model='another'))
    assert backend.embeddings().search_by_vector([1., 0., 0.], application_id=application.id) == []


def test_concurrent_processing_rejected_and_stale_processing_can_retry(processing, backend):
    _, repository, _, _, document, _, command = processing
    with repository.processing_lock(document.id):
        repository.set_status(document.id, 'processing')
        with pytest.raises(ConflictError, match='already'):
            with backend.documents().processing_lock(document.id):
                pytest.fail('Second lease acquired')
        assert repository.get(document.id).status == 'processing'
    assert command().execute().document.status == 'done'
    with pytest.raises(RuntimeError):
        with repository.processing_lock(document.id):
            raise RuntimeError('release on failure')
    assert command().execute().document.status == 'done'


def test_settings_changed_during_embedding_fail_without_replacing(processing, backend):
    application, repository, _, _, document, provider, command = processing
    original = command().execute()
    def embed(texts):
        backend.applications().update(replace(application, data={'chunking': {'chunk_size': 10, 'stride': 10}}))
        return [[1., 0.] for text in texts]
    provider.embed.side_effect = embed
    with pytest.raises(ConflictError, match='changed'):
        command().execute()
    assert repository.get(document.id).status == 'error'
    assert repository.details(document.id).embeddings == original.embeddings


def seed_pre_status(backend):
    create_database(backend.maintenance_url, backend.url.database)
    metadata = build_metadata()
    application = Application(name='old', embedding_model='old', embedding_size=2)
    documents = [Document(name=str(number), application_id=application.id) for number in range(2)]
    chunk = Embedding(document_id=documents[0].id, content='text', embedding=[1., 0.], embedding_model='old', embedding_size=2)
    with backend.engine.begin() as connection:
        connection.exec_driver_sql('CREATE EXTENSION vector')
        metadata.create_all(connection)
        connection.execute(metadata.tables['public.applications'].insert(), asdict(application))
        connection.execute(metadata.tables['public.documents'].insert(), [
            {k: v for k, v in asdict(document).items() if k != 'status'} for document in documents])
        connection.execute(metadata.tables['public.embeddings'].insert(), asdict(chunk))
    return application, documents, chunk


def test_status_migration_backfills_preserves_and_is_idempotent(backend):
    application, documents, chunk = seed_pre_status(backend)
    backend.migrate()
    assert backend.documents().get(documents[0].id).status == 'done'
    assert backend.documents().get(documents[1].id).status == 'pending'
    for document in documents:
        loaded = backend.documents().get(document.id)
        assert loaded.created_at == document.created_at and loaded.updated_at == document.updated_at
        assert SNAPSHOT_KEY in loaded.metadata
    assert backend.embeddings().get(chunk.id).content == chunk.content
    assert backend.embeddings().search_by_vector([1., 0.], application_id=application.id)[0].embedding.id == chunk.id
    before = backend.documents().list()
    backend.migrate()
    assert backend.documents().list() == before


def test_status_upgrade_rejects_unrecognized_schema_without_changes(backend):
    seed_pre_status(backend)
    with backend.engine.begin() as connection:
        connection.exec_driver_sql('ALTER TABLE documents ADD COLUMN unexpected text')
    with pytest.raises(SchemaCompatibilityError):
        backend.migrate()
    assert 'status' not in {c['name'] for c in sa.inspect(backend.engine).get_columns('documents')}


def test_search_compares_preprocessing_positions_exactly(processing, backend):
    application, _, _, _, _, _, command = processing
    command().execute()
    changed = replace(application, data={'provider_config': {'embedding': {
        'document_prefix': 'task: search result | query: ', 'query_prefix': 'title: none | text: '}}})
    backend.applications().update(changed)
    assert backend.embeddings().search_by_vector([1., 0.], application_id=application.id) == []


def test_metadata_update_preserves_processing_snapshot(processing, backend):
    application, repository, _, _, _, _, command = processing
    result = command().execute()
    repository.update(replace(result.document, metadata={'reviewed': True}))
    assert len(backend.embeddings().search_by_vector([1., 0.], application_id=application.id)) == 2
    assert repository.get(result.document.id).metadata['reviewed'] is True


def test_replacement_cannot_forge_document_ownership(processing, backend):
    application, repository, _, _, _, _, command = processing
    original = command().execute()
    other = backend.applications().create(replace(application, id=uuid4(), api_key='other-key', name='other'))
    with pytest.raises(ValueError, match='ownership'):
        repository.replace_embeddings(replace(original.document, application_id=other.id),
            original.embeddings, application=other)
    assert repository.details(original.document.id) == original


def test_status_upgrade_rolls_back_when_later_schema_creation_fails(backend):
    from app.exceptions import DatabaseError
    _, documents, _ = seed_pre_status(backend)
    sa.Table('z_broken', backend.metadata,
        sa.Column('id', sa.Integer, server_default=sa.text('missing_function()')))
    with pytest.raises(DatabaseError):
        backend.migrate()
    assert 'status' not in {c['name'] for c in sa.inspect(backend.engine).get_columns('documents')}
    with backend.engine.connect() as connection:
        assert connection.exec_driver_sql('SELECT count(*) FROM documents').scalar_one() == len(documents)
