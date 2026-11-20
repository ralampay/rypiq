from unittest.mock import ANY
from uuid import uuid4
from unittest.mock import Mock

import pytest

from app.models.document import Document
from app.models.validation import validate_document
from app.exceptions import ValidationError
from app.cli_parser import build_parser


def test_documents_default_to_pending():
    assert Document(name='Report').status == 'pending'


def test_invalid_status_is_rejected():
    document = Document(name='Report', application_id=uuid4())
    document.status = 'unknown'
    with pytest.raises(ValidationError, match='status'):
        validate_document(document)


def test_process_cli_and_status_filter():
    identifier = uuid4()
    args = build_parser().parse_args(['documents:process', '--document-id', str(identifier)])
    assert args.document_id == identifier
    args = build_parser().parse_args(['documents:list', '--application-id', str(identifier), '--status', 'error'])
    assert args.status == 'error'


def test_process_marks_processing_before_provider_and_error_on_failure(tmp_path):
    from contextlib import contextmanager, nullcontext
    from app.models.application import Application
    from app.operations.documents import process
    document = Document(name='Report', application_id=uuid4(), storage_key='file.txt', filename='file.txt')
    repository = Mock()
    repository.processing_lock.return_value = nullcontext()
    repository.set_status.return_value = document
    applications = Mock()
    applications.get.return_value = Application(id=document.application_id, name='App', embedding_model='model', embedding_size=2)
    storage = Mock()
    storage.get.side_effect = lambda key, destination: destination.write_text('Hello')

    @contextmanager
    def provider_scope(application):
        repository.set_status.assert_called_once_with(document.id, 'processing')
        raise RuntimeError('model failed')
        yield

    with pytest.raises(RuntimeError, match='model failed'):
        process.ProcessDocument(document_id=document.id, repository=repository,
            applications=applications, storage=storage, provider_scope=provider_scope).execute()
    assert repository.set_status.call_args.args == (document.id, 'error')
    repository.replace_embeddings.assert_not_called()


@pytest.mark.parametrize('failure_point', ['storage', 'empty', 'provider', 'commit', 'cleanup'])
def test_processing_failures_preserve_original_and_cleanup(tmp_path, document_resources, failure_point):
    from contextlib import contextmanager
    from pathlib import Path
    from app.models.application import Application
    from app.operations.documents.create import CreateDocument
    from app.operations.documents.process import ProcessDocument
    source = tmp_path / 'source.txt'
    source.write_text('hello')
    repository, storage = document_resources(source=source)
    application = Application(name='test', embedding_model='model', embedding_size=2)
    created = CreateDocument(name='Report', path=str(source), application=application,
        repository=repository, storage=storage).execute()
    provider = Mock(model_name='model')
    provider.embed.return_value = [[1., 0.]]
    destinations = []
    def get(key, destination):
        destinations.append(destination)
        if failure_point == 'storage':
            raise OSError('missing stored file')
        destination.write_text('' if failure_point == 'empty' else 'hello')
    storage.get.side_effect = get
    if failure_point == 'provider':
        provider.embed.side_effect = RuntimeError('embedding failed')
    if failure_point == 'commit':
        repository.replace_embeddings.side_effect = RuntimeError('commit failed')
    @contextmanager
    def scope(application):
        yield provider
        if failure_point == 'cleanup':
            raise RuntimeError('provider cleanup failed')
    with pytest.raises((OSError, ValueError, RuntimeError)):
        ProcessDocument(document_id=created.document.id, repository=repository, storage=storage,
            application=application, provider_scope=scope).execute()
    assert repository.set_status.call_args.args == (created.document.id, 'error')
    assert destinations and all(not path.parent.exists() for path in destinations)
    storage.delete.assert_not_called()
    if failure_point != 'commit':
        repository.replace_embeddings.assert_not_called()


def test_creation_inserts_pending_before_storage(tmp_path, document_resources):
    from app.models.application import Application
    from app.operations.documents.create import CreateDocument
    source = tmp_path / 'source.txt'
    source.write_text('hello')
    repository, storage = document_resources(source=source)
    def save(*args, **kwargs):
        repository.create.assert_called_once()
        assert repository.create.call_args.args[0].status == 'pending'
        raise OSError('storage unavailable')
    storage.save.side_effect = save
    with pytest.raises(OSError, match='unavailable') as error:
        CreateDocument(name='Report', path=str(source), application=Application(name='test', embedding_model='model'),
            repository=repository, storage=storage).execute()
    identifier = repository.create.call_args.args[0].id
    assert repository.set_status.call_args.args == (identifier, 'error')
    assert str(identifier) in error.value.__notes__[0]
    repository.delete.assert_not_called()


def test_failure_to_record_error_preserves_primary():
    from app.operations.documents.process import record_failure
    repository = Mock()
    repository.set_status.side_effect = RuntimeError('database down')
    failure = RuntimeError('original')
    record_failure(repository, uuid4(), failure)
    assert str(failure) == 'original'
    assert any('Could not record error status' in note for note in failure.__notes__)


def test_process_cli_dispatch_and_export(monkeypatch, tmp_path, capsys):
    import json
    from app import application, cli
    from app.models.document import DocumentDetails
    document = Document(name='Report', status='done')
    process = Mock(return_value=DocumentDetails(document, []))
    monkeypatch.setattr(application, 'process_document', process)
    output = tmp_path / 'result.json'
    cli.main(['documents:process', '--document-id', str(document.id), '--output', str(output)])
    process.assert_called_once_with(document_id=document.id, batch_size=1, config=ANY)
    assert json.loads(capsys.readouterr().out) == json.loads(output.read_text())
    assert json.loads(output.read_text())['status'] == 'done'


def test_unexpected_cli_processing_failure_reports_retry_id_safely(monkeypatch, capsys):
    from app import application, cli
    identifier = uuid4()
    failure = TypeError('private plugin diagnostic')
    from app.operations.documents.process import record_failure
    record_failure(Mock(), identifier, failure)
    monkeypatch.setattr(application, 'process_document', Mock(side_effect=failure))
    with pytest.raises(SystemExit):
        cli.main(['documents:process', '--document-id', str(identifier)])
    message = capsys.readouterr().err
    assert 'private plugin diagnostic' not in message
    assert f'Document ID: {identifier}' in message
