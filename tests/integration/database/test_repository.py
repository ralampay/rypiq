from dataclasses import asdict
from uuid import UUID, uuid4

import pytest
import sqlalchemy as sa

from app.models.document import Document


def test_crud_metadata_and_stable_pagination(backend):
    backend.create_database()
    repository = backend.documents()
    first = Document(content='first', embedding=[1., 0., 0.], id=UUID(int=1),
                     model_name='example.gguf', metadata={'nested': {'a': [1, 2]}},
                     source_id=uuid4(), storage_key='documents/a.txt')
    second = Document(content='second', embedding=[0., 1., 0.], id=UUID(int=2))
    repository.create(second)
    repository.create(first)
    loaded = repository.get(first.id)
    assert isinstance(loaded.embedding, list)
    assert asdict(loaded) == asdict(first)
    assert [r.id for r in repository.list(1, 0)] == [first.id]
    assert [r.id for r in repository.list(1, 1)] == [second.id]
    first.content = 'updated'
    previous = first.updated_at
    repository.update(first)
    assert repository.get(first.id).content == 'updated'
    assert first.updated_at >= previous
    backend.create_database()
    backend.migrate()
    assert repository.get(first.id).content == 'updated'
    assert repository.delete(first.id) is True
    assert repository.delete(first.id) is False
    assert repository.get(first.id) is None
    with pytest.raises(KeyError):
        repository.update(first)


def test_search_uses_pgvector_sql_and_scalar_filters(backend):
    backend.create_database()
    repository = backend.documents()
    docs = [Document(content='exact', embedding=[1., 0., 0.], model_name='a'),
            Document(content='orthogonal', embedding=[0., 1., 0.], model_name='a'),
            Document(content='opposite', embedding=[-1., 0., 0.], model_name='b')]
    for doc in docs:
        repository.create(doc)
    statements = []
    def capture(conn, cursor, statement, parameters, context, executemany):
        statements.append(statement)
    sa.event.listen(backend.engine, 'before_cursor_execute', capture)
    try:
        matches = repository.search_by_vector([1., 0., 0.], limit=2, filters={'model_name': 'a'})
    finally:
        sa.event.remove(backend.engine, 'before_cursor_execute', capture)
    assert [m.document.content for m in matches] == ['exact', 'orthogonal']
    assert [m.distance for m in matches] == pytest.approx([0., 1.])
    assert any('<=>' in sql and 'ORDER BY' in sql and 'LIMIT' in sql for sql in statements)
    assert repository.search_by_vector([1., 0., 0.], filters={'storage_key': None})
    assert repository.search_by_vector([1., 0., 0.], filters={'model_name': "x' OR TRUE --"}) == []


@pytest.mark.parametrize('vector', [[], [1., 2.], [0., 0., 0.], [float('nan'), 1., 0.],
                                    [float('inf'), 1., 0.], [1e100, 0., 0.], ['x', 1., 0.]])
def test_invalid_vectors_fail_before_database_io(backend, vector):
    repository = backend.documents()  # Target intentionally does not exist.
    with pytest.raises(ValueError, match='[Ee]mbedding'):
        repository.create(Document(content='bad', embedding=vector))
    with pytest.raises(ValueError, match='[Ee]mbedding'):
        repository.update(Document(content='bad', embedding=vector))
    with pytest.raises(ValueError, match='[Ee]mbedding'):
        repository.search_by_vector(vector)


@pytest.mark.parametrize('filters', [{'metadata': {'a': 1}}, {'$and': []}, {'model_name': {'$eq': 'x'}}, []])
def test_unsupported_filters_fail_before_database_io(backend, filters):
    with pytest.raises(ValueError, match='[Ff]ilter|equality'):
        backend.documents().search_by_vector([1., 0., 0.], filters=filters)
