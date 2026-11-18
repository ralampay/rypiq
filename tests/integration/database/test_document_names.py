"""Names, listing counts, and schema upgrades verified on disposable PostgreSQL."""
from dataclasses import asdict, replace
from uuid import UUID, uuid4
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import sqlalchemy as sa

from app.models.application import Application
from app.models.document import Document
from app.models.embedding import Embedding
from app.database.postgresql_connection import create_database


@pytest.fixture
def application(backend):
    backend.create_database()
    return backend.applications().create(Application(name='a', embedding_model='m', embedding_size=2))


def test_names_unique_per_application_and_renames(backend, application):
    repo = backend.documents()
    first = repo.create(Document(name=' Report ', application_id=application.id))
    assert first.name == repo.get(first.id).name == 'Report'
    with pytest.raises(ValueError, match='already exists'):
        repo.create(Document(name='Report', application_id=application.id))
    second = repo.create(Document(name='report', application_id=application.id))
    other = backend.applications().create(Application(name='other', embedding_model='m', embedding_size=2))
    repo.create(Document(name='Report', application_id=other.id))
    assert repo.name_exists(application_id=application.id, name='Report') is True
    with pytest.raises(ValueError, match='already exists'):
        repo.update(replace(second, name='Report'))
    assert repo.get(second.id).name == 'report'
    repo.update(replace(second, name=' renamed '))
    assert repo.get(second.id).name == 'renamed'


@pytest.mark.parametrize('name', ['Report', '', None])
def test_database_enforces_names_even_for_direct_sql(backend, application, name):
    backend.documents().create(Document(name='Report', application_id=application.id))
    value = asdict(Document(name=name, application_id=application.id))
    with pytest.raises(sa.exc.IntegrityError):
        with backend.engine.begin() as connection:
            connection.execute(backend.metadata.tables['public.documents'].insert().values(**value))


def test_concurrent_creations_have_one_winner(backend, application):
    barrier = Barrier(2)
    def create():
        repository = backend.documents()
        assert not repository.name_exists(application_id=application.id, name='same')
        barrier.wait(timeout=10)
        try:
            return repository.create(Document(name='same', application_id=application.id))
        except ValueError as error:
            return str(error)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: create(), range(2)))
    assert sum(isinstance(result, Document) for result in results) == 1
    assert sum(isinstance(result, str) and 'already exists' in result for result in results) == 1
    assert len(backend.documents().list(application_id=application.id)) == 1


def test_name_race_compensates_storage_and_rolls_back_chunks(backend, application, tmp_path):
    from app.operations.documents.ingest import IngestDocument
    from app.storage.providers.local import LocalStorage

    repository = backend.documents()
    storage = LocalStorage(tmp_path / 'objects')
    source = tmp_path / 'source.txt'
    source.write_text('hello')
    class RacingProvider:
        model_name = 'm'
        def embed(self, texts):
            # Another ingestion wins after the preflight but before our atomic write.
            repository.create(Document(name='same', application_id=application.id))
            return [[1., 0.]]
    with pytest.raises(ValueError, match='already exists'):
        IngestDocument(name='same', path=str(source), application=application,
                       chunk_size=10, stride=10, embedding_provider=RacingProvider(),
                       repository=repository, storage=storage).execute()
    assert storage.list() == []
    parents = repository.list(application_id=application.id)
    assert len(parents) == 1
    assert backend.embeddings().list(document_id=parents[0].id) == []


def test_listing_is_scoped_ordered_and_counts_without_loading_vectors(backend, application):
    from app.operations.documents.list import list_documents

    repository = backend.documents()
    parents = [Document(name=f'doc {number}', id=UUID(int=number + 1), application_id=application.id)
               for number in range(101)]
    for parent in reversed(parents):
        repository.create(parent)
    chunk = Embedding(document_id=parents[0].id, content='text', embedding=[1., 0.],
                      embedding_model='m', embedding_size=2)
    backend.embeddings().create(chunk)
    backend.embeddings().create(replace(chunk, id=uuid4()))
    other = backend.applications().create(Application(name='other', embedding_model='m', embedding_size=2))
    repository.create(Document(name='outside', application_id=other.id))
    statements = []
    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)
    sa.event.listen(backend.engine, 'before_cursor_execute', capture)
    try:
        result = list_documents(application_id=application.id, repository=repository,
                                applications=backend.applications())
    finally:
        sa.event.remove(backend.engine, 'before_cursor_execute', capture)
    assert [row.id for row in result] == [parent.id for parent in parents]
    assert [row.embedding_count for row in result] == [2] + [0] * 100
    assert len(statements) == 3  # One application read and two summary pages.
    assert all('embeddings.embedding,' not in statement for statement in statements)
    backend.embeddings().delete(chunk.id)
    assert repository.list_summaries(application_id=application.id)[0].embedding_count == 1


def seed_unnamed(backend):
    from app.database.unnamed_schema import build_metadata
    create_database(backend.maintenance_url, backend.url.database)
    old = build_metadata()
    application = Application(name='a', embedding_model='m', embedding_size=2)
    documents = [Document(name='unused', id=UUID(int=i + 1), application_id=application.id,
                          filename=filename, metadata={'keep': True})
                 for i, filename in enumerate(['same', 'same', '', f'same ({UUID(int=1)})',
                                               f'Document {UUID(int=3)}', ' unique '])]
    with backend.engine.begin() as connection:
        connection.exec_driver_sql('CREATE EXTENSION vector')
        old.create_all(connection)
        connection.execute(old.tables['public.applications'].insert().values(**asdict(application)))
        connection.execute(old.tables['public.documents'].insert(), [
            {key: value for key, value in asdict(document).items() if key != 'name'} for document in documents])
        chunk = Embedding(document_id=documents[0].id, content='keep', embedding=[1., 0.],
                          embedding_model='m', embedding_size=2)
        connection.execute(old.tables['public.embeddings'].insert().values(**asdict(chunk)))
    return application, documents, chunk


def test_name_migration_preserves_values_and_handles_generated_collisions(backend):
    application, originals, chunk = seed_unnamed(backend)
    assert backend.migrate() == {'tables_created': []}
    loaded = backend.documents().list(application_id=application.id)
    assert [document.name for document in loaded] == [
        f'same ({UUID(int=1)}) (2)', f'same ({UUID(int=2)})',
        f'Document {UUID(int=3)} (2)', f'same ({UUID(int=1)})',
        f'Document {UUID(int=3)}', 'unique']
    for before, after in zip(originals, loaded, strict=True):
        assert asdict(replace(after, name=before.name)) == asdict(before)
    assert backend.embeddings().get(chunk.id) == chunk
    assert backend.migrate() == {'tables_created': []}
    assert backend.documents().list(application_id=application.id) == loaded


def test_name_migration_failure_rolls_back_column_and_data(backend):
    _, originals, _ = seed_unnamed(backend)
    sa.Table('z_broken', backend.metadata, sa.Column('id', sa.Integer, server_default=sa.text('missing_function()')))
    with pytest.raises(sa.exc.ProgrammingError):
        backend.migrate()
    assert 'name' not in {column['name'] for column in sa.inspect(backend.engine).get_columns('documents')}
    with backend.engine.connect() as connection:
        assert connection.exec_driver_sql('SELECT count(*) FROM documents').scalar_one() == len(originals)


@pytest.mark.parametrize('constraint', ['uq_documents_application_name', 'ck_documents_name_nonempty'])
def test_name_constraint_drift_rejected(backend, application, constraint):
    with backend.engine.begin() as connection:
        connection.exec_driver_sql(f'ALTER TABLE documents DROP CONSTRAINT {constraint}')
    with pytest.raises(ValueError, match='Incompatible'):
        backend.migrate()
