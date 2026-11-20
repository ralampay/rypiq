"""Provider split migration and update compatibility against disposable PostgreSQL."""
from dataclasses import asdict, replace

import pytest
import sqlalchemy as sa

from app import application as entrypoints
from app.database.postgresql_connection import create_database
from app.database.shared_provider_schema import build_metadata
from app.exceptions import SchemaCompatibilityError, ValidationError
from app.models.application import Application
from app.models.document import Document
from app.models.embedding import Embedding
from app.models.processing import processing_settings, SNAPSHOT_KEY


def seed_shared(backend, provider='local'):
    create_database(backend.maintenance_url, backend.url.database)
    metadata = build_metadata()
    value = Application(name='existing', embedding_model='embed', inference_model='chat',
                        embedding_size=2, embedding_provision_type=provider,
                        inference_provision_type=provider, data={'keep': True})
    parent = Document(name='Guide', application_id=value.id, status='done',
                      metadata={SNAPSHOT_KEY: processing_settings(value)})
    chunk = Embedding(document_id=parent.id, content='Preserve this', embedding=[1., 0.],
                      embedding_model='embed', model_name='embed', embedding_size=2, provision_type=provider)
    row = asdict(value)
    row['provision_type'] = row.pop('embedding_provision_type')
    row.pop('inference_provision_type')
    with backend.engine.begin() as connection:
        connection.exec_driver_sql('CREATE EXTENSION vector')
        metadata.create_all(connection)
        connection.execute(metadata.tables['public.applications'].insert(), row)
        connection.execute(metadata.tables['public.documents'].insert(), asdict(parent))
        connection.execute(metadata.tables['public.embeddings'].insert(), asdict(chunk))
    return value, parent, chunk


@pytest.mark.parametrize('provider', ['local', 'openai', 'llama-server'])
def test_split_preserves_records_and_inference_updates_do_not_invalidate_search(backend, provider):
    original, parent, chunk = seed_shared(backend, provider)
    backend.migrate()
    assert backend.applications().get(original.id) == original
    assert backend.documents().get(parent.id) == parent
    assert backend.embeddings().get(chunk.id) == chunk
    assert backend.migrate() == {'tables_created': []}
    updated = backend.applications().update(replace(original, inference_provision_type='openai', inference_model='new-chat'))
    assert updated.embedding_provision_type == provider
    assert [r.embedding.id for r in backend.embeddings().search_by_vector([1., 0.], application_id=original.id)] == [chunk.id]
    backend.applications().update(replace(updated, embedding_provision_type='openai' if provider == 'local' else 'local'))
    assert backend.embeddings().search_by_vector([1., 0.], application_id=original.id) == []


def test_unrecognized_schema_does_not_partially_migrate(backend):
    seed_shared(backend)
    with backend.engine.begin() as connection:
        connection.exec_driver_sql('ALTER TABLE applications ADD COLUMN inference_provision_type text')
    with pytest.raises(SchemaCompatibilityError):
        backend.migrate()
    columns = {c['name'] for c in sa.inspect(backend.engine).get_columns('applications')}
    assert 'provision_type' in columns and 'embedding_provision_type' not in columns


def test_later_failure_rolls_back_provider_split(backend):
    seed_shared(backend)
    sa.Table('z_failure', backend.metadata, sa.Column('id', sa.Integer, server_default=sa.text('missing_function()')))
    from app.exceptions import DatabaseError
    with pytest.raises(DatabaseError):
        backend.migrate()
    columns = {c['name'] for c in sa.inspect(backend.engine).get_columns('applications')}
    assert 'provision_type' in columns and 'embedding_provision_type' not in columns


def test_invalid_update_preserves_persisted_configuration(backend, monkeypatch):
    backend.create_database()
    value = backend.applications().create(Application(name='existing', embedding_model='embed'))
    monkeypatch.setattr(entrypoints, 'get_database', lambda config=None: backend)
    with pytest.raises(ValidationError):
        entrypoints.update_application(application=replace(value, inference_provision_type='unknown'))
    assert backend.applications().get(value.id) == value
    with pytest.raises(sa.exc.IntegrityError):
        with backend.engine.begin() as connection:
            connection.execute(backend.metadata.tables['public.applications'].update().values(inference_provision_type='unknown'))
    assert backend.applications().get(value.id) == value


def test_cli_updates_existing_application_after_migration(backend, monkeypatch, capsys):
    import json
    from app import cli
    original, parent, chunk = seed_shared(backend)
    backend.migrate()
    monkeypatch.setattr(entrypoints, 'get_database', lambda config=None: backend)
    cli.main(['applications:update', str(original.id), '--inference-provision-type', 'openai',
              '--inference-model', 'remote-chat'])
    result = json.loads(capsys.readouterr().out)
    assert result['embedding_provision_type'] == 'local'
    assert result['inference_provision_type'] == 'openai'
    assert 'api_key' not in result and 'provision_type' not in result
    saved = backend.applications().get(original.id)
    assert saved.inference_model == 'remote-chat'
    assert saved.embedding_model == original.embedding_model
    assert saved.api_key == original.api_key and saved.created_at == original.created_at
    assert saved.data == original.data
    assert backend.documents().get(parent.id) == parent
    assert [r.embedding.id for r in backend.embeddings().search_by_vector([1., 0.], application_id=original.id)] == [chunk.id]
    cli.main(['applications:update', str(original.id), '--data', '{"new": true}', '--clear-inference-model'])
    saved = backend.applications().get(original.id)
    assert saved.data == {'new': True} and saved.inference_model is None
    assert saved.inference_provision_type == 'openai'
