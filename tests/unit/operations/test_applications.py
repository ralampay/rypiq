from unittest.mock import Mock

import pytest

from app.models.application import Application
from app.operations.applications.create import CreateApplication
from app.operations.applications.get import get_application
from app.operations.documents.ingest import IngestDocument
from app.operations.documents.search import SearchDocuments


@pytest.mark.parametrize('changes', [
    {'name': ''}, {'embedding_model': ' '}, {'embedding_size': True},
    {'embedding_size': 0}, {'embedding_size': 16001}, {'provision_type': 'unknown'},
    {'data': []}, {'data': {'invalid': float('nan')}},
])
def test_invalid_application_never_writes(changes):
    values = dict(name='test', embedding_model='model', embedding_size=3)
    values.update(changes)
    repository = Mock()
    command = CreateApplication(application=Application(**values), repository=repository)
    repository.create.assert_not_called()
    with pytest.raises(ValueError):
        command.execute()
    repository.create.assert_not_called()


def test_create_returns_repository_result_and_missing_application_raises():
    repository = Mock()
    application = Application(name='test', embedding_model='model')
    repository.create.return_value = application
    assert CreateApplication(application=application, repository=repository).execute() is application
    repository.get.return_value = None
    with pytest.raises(KeyError):
        get_application(application_id=application.id, repository=repository)


@pytest.mark.parametrize('provider_type', ['openai', 'llama-server'])
def test_remote_provider_is_borrowed_and_inputs_are_unprefixed(tmp_path, provider_type):
    source = tmp_path / 'input.txt'
    source.write_text('hello')
    application = Application(name='test', embedding_model='remote/model', embedding_size=2,
                              provision_type=provider_type)
    provider, repository, storage = Mock(model_name='remote/model'), Mock(), Mock()
    provider.embed.return_value = [[1., 0.]]
    storage.save.return_value = Mock(key='stored', content_type='text/plain', checksum='checksum')
    documents = IngestDocument(path=str(source), chunk_size=10, stride=10, application=application,
                               embedding_provider=provider, repository=repository, storage=storage).execute()
    assert documents[0].embedding_model == 'remote/model'
    provider.embed.assert_called_once_with(['hello'])
    provider.embed.reset_mock()
    SearchDocuments(query='query', application=application, embedding_provider=provider,
                    repository=repository).execute()
    provider.embed.assert_called_once_with(['query'])
    provider.close.assert_not_called()


@pytest.mark.parametrize('vector', [[1., 0.], [0., 0., 0.], [1e100, 0., 0.]])
def test_bad_application_vector_rejected_before_storage(tmp_path, vector):
    source = tmp_path / 'file.txt'
    source.write_text('hello')
    application = Application(name='test', embedding_model='model', embedding_size=3)
    provider = Mock(model_name='model')
    provider.embed.return_value = [vector]
    storage = Mock()
    with pytest.raises(ValueError, match='Embedding'):
        IngestDocument(path=str(source), chunk_size=10, stride=10, application=application,
                       embedding_provider=provider, repository=Mock(), storage=storage).execute()
    storage.save.assert_not_called()
