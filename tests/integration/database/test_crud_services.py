from dataclasses import replace
from uuid import uuid4

import pytest

from app.models.application import Application
from app.models.document import Document
from app.services.application import ApplicationService
from app.services.document import DocumentService


@pytest.fixture
def services(backend):
    backend.create_database()
    applications = ApplicationService(repository=backend.applications())
    first = applications.create(Application(name='first', embedding_model='a', embedding_size=2))
    second = applications.create(Application(name='second', embedding_model='b', embedding_size=3))
    documents = DocumentService(application_id=first.id, repository=backend.documents(),
                                applications=backend.applications())
    return applications, documents, first, second


def test_application_update_delete_and_missing_records(services):
    applications, _, first, _ = services
    changed = replace(first, name='renamed', data={'nested': [1]}, inference_model='inference',
                      embedding_model='new', embedding_size=4, provision_type='openai')
    previous = changed.updated_at
    updated = applications.update(changed)
    assert updated is changed
    assert updated.updated_at >= previous
    loaded = applications.get(first.id)
    assert loaded.name == 'renamed'
    assert loaded.data == {'nested': [1]}
    assert loaded.inference_model == 'inference'
    assert loaded.embedding_size == 4
    assert loaded.api_key == first.api_key
    assert loaded.created_at == first.created_at
    assert applications.delete(first.id) is True
    assert applications.delete(first.id) is False
    with pytest.raises(KeyError):
        applications.update(changed)
    with pytest.raises(KeyError):
        applications.get(first.id)


def test_application_embedding_changes_and_delete_restricted_with_documents(services):
    applications, documents, first, _ = services
    created = documents.create(Document(filename='text', storage_key='shared.txt', name=str(uuid4())))
    assert applications.update(replace(first, name='allowed')).name == 'allowed'
    for changes in ({'embedding_size': 3}, {'embedding_model': 'other'}, {'provision_type': 'openai'}):
        with pytest.raises(ValueError, match='documents'):
            applications.update(replace(first, **changes))
        assert applications.get(first.id).embedding_model == first.embedding_model
    with pytest.raises(ValueError, match='documents'):
        applications.delete(first.id)
    assert documents.get(created.id).storage_key == 'shared.txt'
    assert documents.delete(created.id) is True
    assert applications.delete(first.id) is True


def test_application_identity_fields_cannot_be_changed(services):
    applications, _, first, _ = services
    for changes in ({'api_key': 'replacement'}, {'created_at': first.created_at.replace(year=2000)}):
        with pytest.raises(ValueError, match='immutable'):
            applications.update(replace(first, **changes))
    assert applications.get(first.id).api_key == first.api_key


def test_document_crud_is_application_scoped_and_preserves_storage(services, backend, tmp_path):
    _, documents, first, second = services
    other = DocumentService(application_id=second.id, repository=backend.documents(),
                            applications=backend.applications())
    source = tmp_path / 'shared.txt'
    source.write_text('keep')
    a = documents.create(Document(filename='a', storage_key=str(source), name=str(uuid4())))
    b = other.create(Document(filename='b', name=str(uuid4())))
    assert [item.id for item in documents.list()] == [a.id]
    assert [item.id for item in other.list()] == [b.id]
    assert documents.list(offset=1) == []
    with pytest.raises(KeyError):
        documents.get(b.id)
    assert documents.delete(b.id) is False
    # Forged ownership still cannot update a row outside the SQL scope.
    with pytest.raises(KeyError):
        documents.update(replace(b, application_id=first.id))
    changed = replace(a, filename='updated', metadata={'extra': 1})
    assert documents.update(changed).filename == 'updated'
    assert documents.get(a.id).metadata == {'extra': 1}
    with pytest.raises(ValueError):
        documents.update(replace(a, application_id=second.id))
    assert documents.delete(a.id) is True
    assert documents.delete(a.id) is False
    assert source.read_text() == 'keep'
    assert other.get(b.id).filename == 'b'
    with pytest.raises(KeyError):
        documents.update(a)


def test_repository_scope_cannot_reassign_documents(services, backend):
    _, documents, first, second = services
    a = documents.create(Document(filename='text', name=str(uuid4())))
    with pytest.raises(ValueError, match='scope'):
        backend.documents().update(a, application_id=second.id)
    # Even unscoped legacy repository updates keep immutable ownership.
    with pytest.raises(ValueError, match='ownership'):
        backend.documents().update(replace(a, application_id=second.id))
    assert documents.get(a.id).application_id == first.id


@pytest.mark.parametrize('action', ['update', 'delete'])
def test_application_mutation_locks_out_concurrent_document_creation(services, backend, monkeypatch, action):
    import sqlalchemy as sa
    from app.database.repositories.postgresql.document import PostgreSQLDocumentRepository

    applications, _, first, _ = services
    repository = applications.repository
    original_check = repository._has_documents
    writer_engine = sa.create_engine(backend.url, connect_args={'options': '-c lock_timeout=200ms'})
    writer = PostgreSQLDocumentRepository(writer_engine, backend.metadata.tables['public.documents'])
    attempts = []

    def check_while_writer_attempts(connection, application_id):
        # A second connection attempts creation while the real mutation holds its lock.
        candidate = Document(filename='concurrent', application_id=first.id, name=str(uuid4()))
        with pytest.raises(sa.exc.OperationalError) as caught:
            writer.create(candidate)
        assert caught.value.orig.sqlstate == '55P03'  # lock_not_available
        attempts.append(candidate.id)
        return original_check(connection, application_id)

    monkeypatch.setattr(repository, '_has_documents', check_while_writer_attempts)
    try:
        if action == 'update':
            applications.update(replace(first, embedding_model='new'))
        else:
            applications.delete(first.id)
        assert len(attempts) == 1
        assert backend.documents().get(attempts[0]) is None
    finally:
        writer_engine.dispose()


def test_application_rejected_update_preserves_timestamp_and_data(services):
    applications, documents, first, _ = services
    documents.create(Document(filename='text', name=str(uuid4())))
    candidate = replace(first, embedding_size=3, data={'uncommitted': True})
    with pytest.raises(ValueError):
        applications.update(candidate)
    loaded = applications.get(first.id)
    assert loaded.updated_at == first.updated_at == candidate.updated_at
    assert loaded.data == first.data == {}


@pytest.mark.parametrize('changes', [{'document_prefix': 'new'}, {'query_prefix': 'new'}, {'dimensions': 2}])
def test_embedding_preprocessing_cannot_change_with_documents(services, changes):
    applications, documents, first, _ = services
    documents.create(Document(filename='text', name=str(uuid4())))
    with pytest.raises(ValueError, match='documents'):
        applications.update(replace(first, data={'provider_config': {'embedding': changes}}))
    # Inference settings and endpoint/credential locations do not change embedding preprocessing.
    changed = applications.update(replace(first, data={'provider_config': {
        'inference': {'context_size': 4096}, 'embedding': {'api_key_env': 'NEW_KEY'}}}))
    assert changed.data['provider_config']['inference']['context_size'] == 4096
