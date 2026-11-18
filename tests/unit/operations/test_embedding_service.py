from dataclasses import replace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.models.embedding import Embedding
from app.services.embedding import EmbeddingService


def test_chunk_crud_scopes_every_call_and_borrows_resources():
    repository = Mock()
    app_id = uuid4()
    service = EmbeddingService(application_id=app_id, repository=repository)
    chunk = Embedding(document_id=uuid4(), content='text', embedding=[1., 0.], embedding_size=2)
    repository.create.return_value = repository.get.return_value = repository.update.return_value = chunk
    repository.list.return_value = [chunk]
    repository.delete.return_value = True
    assert service.create(chunk) is chunk
    assert service.get(chunk.id) is chunk
    assert service.list(document_id=chunk.document_id) == [chunk]
    assert service.update(chunk) is chunk
    assert service.delete(chunk.id) is True
    for method in ('create', 'get', 'list', 'update', 'delete'):
        assert getattr(repository, method).call_args.kwargs['application_id'] == app_id
    repository.close.assert_not_called()
    repository.get.return_value = None
    with pytest.raises(KeyError):
        service.get(chunk.id)


@pytest.mark.parametrize('changes', [
    {'content': None}, {'metadata': []}, {'metadata': {'bad': float('nan')}},
    {'embedding': [0., 0.]}, {'embedding': [1.]}, {'id': 'invalid'},
    {'document_id': None}, {'chunk_index': -1}, {'chunk_index': True},
])
@pytest.mark.parametrize('method', ['create', 'update'])
def test_invalid_chunks_fail_without_io(method, changes):
    repository = Mock()
    service = EmbeddingService(application_id=uuid4(), repository=repository)
    chunk = Embedding(document_id=uuid4(), content='text', embedding=[1., 0.], embedding_size=2)
    with pytest.raises(ValueError):
        getattr(service, method)(replace(chunk, **changes))
    assert not repository.mock_calls
