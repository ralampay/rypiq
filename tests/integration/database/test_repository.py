from dataclasses import asdict
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa

from app.models.document import Document
from app.models.embedding import Embedding
from app.models.application import Application


@pytest.fixture
def application(backend):
    backend.create_database()
    return backend.applications().create(Application(name="test", embedding_model="example.gguf", embedding_size=3))


def test_crud_metadata_and_stable_pagination(backend, application):
    backend.create_database()
    repository = backend.documents()
    first = Document(application_id=application.id, id=UUID(int=1), filename='first',
                     metadata={'nested': {'a': [1, 2]}}, storage_key='documents/a.txt')
    second = Document(application_id=application.id, id=UUID(int=2), filename='second')
    repository.create(second)
    repository.create(first)
    loaded = repository.get(first.id)
    assert asdict(loaded) == asdict(first)
    assert [r.id for r in repository.list(1, 0)] == [first.id]
    assert [r.id for r in repository.list(1, 1)] == [second.id]
    first.filename = 'updated'
    previous = first.updated_at
    repository.update(first)
    assert repository.get(first.id).filename == 'updated'
    assert first.updated_at >= previous
    backend.create_database()
    backend.migrate()
    assert repository.get(first.id).filename == 'updated'
    assert repository.delete(first.id) is True
    assert repository.delete(first.id) is False
    assert repository.get(first.id) is None
    with pytest.raises(KeyError):
        repository.update(first)


def test_search_uses_pgvector_sql_and_scalar_filters(backend, application):
    backend.create_database()
    repository = backend.embeddings()
    parent = backend.documents().create(Document(application_id=application.id))
    docs = [Embedding(document_id=parent.id, content='exact', embedding=[1., 0., 0.], model_name='a'),
            Embedding(document_id=parent.id, content='orthogonal', embedding=[0., 1., 0.], model_name='a'),
            Embedding(document_id=parent.id, content='opposite', embedding=[-1., 0., 0.], model_name='b')]
    for doc in docs:
        doc.embedding_model = application.embedding_model
        doc.embedding_size = 3
        repository.create(doc)
    statements = []
    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)
    sa.event.listen(backend.engine, 'before_cursor_execute', capture)
    try:
        matches = repository.search_by_vector([1., 0., 0.], application_id=application.id, limit=2, filters={'model_name': 'a'})
    finally:
        sa.event.remove(backend.engine, 'before_cursor_execute', capture)
    assert [m.embedding.content for m in matches] == ['exact', 'orthogonal']
    assert [m.distance for m in matches] == pytest.approx([0., 1.])
    assert any('<=>' in sql and 'ORDER BY' in sql and 'LIMIT' in sql for sql in statements)
    assert repository.search_by_vector([1., 0., 0.], application_id=application.id, filters={'storage_key': None})
    assert repository.search_by_vector([1., 0., 0.], application_id=application.id, filters={'model_name': "x' OR TRUE --"}) == []


@pytest.mark.parametrize('vector', [[], [1., 2.], [0., 0., 0.], [float('nan'), 1., 0.],
                                    [float('inf'), 1., 0.], [1e100, 0., 0.], ['x', 1., 0.]])
def test_invalid_vectors_are_rejected(backend, application, vector):
    repository = backend.embeddings()
    parent = Document(application_id=application.id)
    with pytest.raises(ValueError, match='[Ee]mbedding'):
        repository.create(Embedding(document_id=parent.id, content='bad', embedding=vector, embedding_size=3))
    with pytest.raises(ValueError, match='[Ee]mbedding'):
        repository.update(Embedding(document_id=parent.id, content='bad', embedding=vector, embedding_size=3))
    with pytest.raises(ValueError, match='[Ee]mbedding'):
        repository.search_by_vector(vector, application_id=application.id)


@pytest.mark.parametrize('filters', [{'metadata': {'a': 1}}, {'$and': []}, {'model_name': {'$eq': 'x'}}, []])
def test_unsupported_filters_fail_before_database_io(backend, filters):
    with pytest.raises(ValueError, match='[Ff]ilter|equality'):
        backend.embeddings().search_by_vector([1., 0., 0.], application_id=uuid4(), filters=filters)
