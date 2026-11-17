from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.operations import document, embedding, search
from app.models.document import Document
from app.models.search import SearchResult


def provider():
    return Mock(model_name='fake.gguf', embed=Mock(side_effect=lambda texts: [[1., 0., 0.] for _ in texts]))


def test_embedding_command_borrows_provider_and_defers_io(tmp_path):
    path = tmp_path / 'not-yet-created.txt'
    model = provider()
    command = embedding.EmbedTextFile(path=str(path), chunk_size=4, stride=4,
                                      embedding_provider=model, batch_size=2)
    model.embed.assert_not_called()
    path.write_text('ABCDEFGH')
    result = command.execute()
    assert result['model'] == 'fake.gguf'
    assert result['embedded_chunk_count'] == 2
    model.embed.assert_called_once()
    model.close.assert_not_called()


def test_search_command_returns_domain_values_without_output(capsys):
    model = provider()
    repository = Mock()
    matches = [SearchResult(Document(content='found', embedding=[1., 0., 0.]), 0.)]
    repository.search_by_vector.return_value = matches
    command = search.SearchDocuments(query='find', embedding_provider=model, repository=repository, limit=2)
    model.embed.assert_not_called()
    assert command.execute() == matches
    assert capsys.readouterr().out == ''
    model.close.assert_not_called()
    repository.close.assert_not_called()


def test_ingestion_command_uses_injected_dependencies(tmp_path):
    path = tmp_path / 'source.txt'
    path.write_text('ABCDEFGH')
    model = provider()
    storage = Mock()
    storage.save.return_value = SimpleNamespace(key='documents/source.txt', content_type='text/plain', checksum='checksum')
    repository = Mock()
    command = document.IngestDocument(path=str(path), storage=storage, repository=repository,
                                      embedding_provider=model, chunk_size=4, stride=4)
    storage.save.assert_not_called()
    result = command.execute()
    assert [item.content for item in result] == ['ABCD', 'EFGH']
    assert all(item.model_name == 'fake.gguf' for item in result)
    assert result[0].source_id == result[1].source_id
    assert repository.create.call_count == 2
    model.close.assert_not_called()
    storage.close.assert_not_called()


def test_ingestion_preserves_failure_and_attempts_all_compensation(tmp_path):
    path = tmp_path / 'source.txt'
    path.write_text('ABCDEFGHIJKL')
    storage = Mock()
    storage.save.return_value = SimpleNamespace(key='owned', content_type='text/plain', checksum='checksum')
    primary = RuntimeError('write failure')
    repository = Mock()
    repository.create.side_effect = [None, None, primary]
    repository.delete.side_effect = OSError('cleanup failure')
    command = document.IngestDocument(path=str(path), storage=storage, repository=repository,
                                      embedding_provider=provider(), chunk_size=4, stride=4)
    with pytest.raises(RuntimeError, match='write failure') as error:
        command.execute()
    assert error.value is primary
    assert repository.delete.call_count == 2
    storage.delete.assert_called_once_with('owned')
    assert len(primary.__notes__) == 2


def test_embedding_validation_finishes_before_storage(tmp_path):
    path = tmp_path / 'source.txt'
    path.write_text('ABCD')
    model = provider()
    model.embed.side_effect = None
    model.embed.return_value = [[float('nan')]]
    storage = Mock()
    command = document.IngestDocument(path=str(path), storage=storage, repository=Mock(),
                                      embedding_provider=model, chunk_size=4, stride=4)
    with pytest.raises(ValueError, match='invalid'):
        command.execute()
    storage.save.assert_not_called()


@pytest.mark.parametrize('vectors, message', [([[]], 'Empty'), ([[1.], [1., 2.]], 'Inconsistent')])
def test_embedding_command_rejects_invalid_provider_results(tmp_path, vectors, message):
    path = tmp_path / 'file.txt'
    path.write_text('ABCD' if len(vectors) == 1 else 'ABCDEFGH')
    model = provider()
    model.embed.side_effect = None
    model.embed.return_value = vectors
    with pytest.raises(ValueError, match=message):
        embedding.EmbedTextFile(path=str(path), chunk_size=4, stride=4, batch_size=2,
                               embedding_provider=model).execute()


def test_search_rejects_nonfinite_provider_output():
    model = provider()
    model.embed.side_effect = None
    model.embed.return_value = [[float('inf')]]
    repository = Mock()
    with pytest.raises(ValueError, match='invalid'):
        search.SearchDocuments(query='hello', embedding_provider=model, repository=repository).execute()
    repository.search_by_vector.assert_not_called()


def test_ingestion_preserves_primary_when_storage_compensation_fails(tmp_path):
    path = tmp_path / 'source.txt'
    path.write_text('ABCD')
    storage = Mock()
    storage.save.return_value = SimpleNamespace(key='owned', content_type='text/plain', checksum='checksum')
    storage.delete.side_effect = OSError('cleanup failure')
    repository = Mock()
    repository.create.side_effect = ValueError('primary')
    with pytest.raises(ValueError, match='primary') as error:
        document.IngestDocument(path=str(path), chunk_size=4, stride=4,
                                embedding_provider=provider(), storage=storage,
                                repository=repository).execute()
    assert 'stored source' in error.value.__notes__[0]
