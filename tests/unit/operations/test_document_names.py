from unittest.mock import Mock
from uuid import uuid4

import pytest

from app import cli
from app.database.schema import build_metadata
from app.models.document import Document
from app.models.validation import validate_document


def test_document_name_is_persisted_and_unique_within_application():
    table = build_metadata().tables['public.documents']
    assert 'name' in table.c
    assert not table.c.name.nullable
    assert any(constraint.name == 'uq_documents_application_name' for constraint in table.constraints)


def test_document_name_is_trimmed():
    document = Document(name='  Annual report  ', application_id=uuid4())
    validate_document(document)
    assert document.name == 'Annual report'


def test_documents_list_json_dispatch(monkeypatch, capsys):
    from app import application
    listing = Mock(return_value=[])
    monkeypatch.setattr(application, 'enumerate_documents', listing, raising=False)
    identifier = uuid4()
    cli.main(['documents:list', '--application-id', str(identifier), '--output', 'json'])
    assert capsys.readouterr().out.strip() == '[]'
    listing.assert_called_once_with(application_id=identifier)


@pytest.mark.parametrize('name', ['', ' \t\n ', None, 42])
def test_invalid_name_prevents_ingestion_io(name):
    from app.models.application import Application
    from app.operations.documents.ingest import IngestDocument
    repository, provider, storage = Mock(), Mock(model_name='m'), Mock()
    command = IngestDocument(name=name, path='unused', chunk_size=5, stride=5,
                             application=Application(name='a', embedding_model='m'),
                             repository=repository, embedding_provider=provider, storage=storage)
    assert not repository.mock_calls
    with pytest.raises(ValueError, match='name'):
        command.execute()
    repository.name_exists.assert_not_called()
    provider.embed.assert_not_called()
    storage.save.assert_not_called()


def test_duplicate_name_rejected_before_embedding_or_storage():
    from app.models.application import Application
    from app.operations.documents.ingest import IngestDocument
    repository = Mock(name_exists=Mock(return_value=True))
    provider, storage = Mock(model_name='m'), Mock()
    app = Application(name='a', embedding_model='m')
    with pytest.raises(ValueError, match='already exists'):
        IngestDocument(name=' report ', path='unused', chunk_size=5, stride=5,
                       application=app, repository=repository,
                       embedding_provider=provider, storage=storage).execute()
    repository.name_exists.assert_called_once_with(application_id=app.id, name='report')
    provider.embed.assert_not_called()
    storage.save.assert_not_called()


def test_listing_pages_and_missing_application():
    from app.models.document import DocumentSummary
    from app.operations.documents.list import list_documents
    repository, applications = Mock(), Mock()
    identifier = uuid4()
    rows = [DocumentSummary(id=uuid4(), name=str(index), embedding_count=index) for index in range(101)]
    repository.list_summaries.side_effect = [rows[:100], rows[100:]]
    assert list_documents(application_id=identifier, repository=repository, applications=applications) == rows
    assert [call.kwargs for call in repository.list_summaries.call_args_list] == [
        dict(application_id=identifier, limit=100, offset=0),
        dict(application_id=identifier, limit=100, offset=100)]
    applications.get.assert_called_once_with(identifier)
    repository.close.assert_not_called()
    applications.get.return_value = None
    repository.reset_mock()
    with pytest.raises(KeyError):
        list_documents(application_id=identifier, repository=repository, applications=applications)
    repository.list_summaries.assert_not_called()


@pytest.mark.parametrize('arguments', [[], ['--application-id', 'bad'],
                                      ['--application-id', str(uuid4()), '--limit', '5']])
def test_listing_rejects_invalid_cli_arguments(arguments, monkeypatch):
    from app import application
    database = Mock()
    monkeypatch.setattr(application, 'get_database', database)
    with pytest.raises(SystemExit):
        cli.main(['documents:list', *arguments])
    database.assert_not_called()


def test_creation_requires_name_before_io(monkeypatch):
    from app import application
    database = Mock()
    monkeypatch.setattr(application, 'get_database', database)
    with pytest.raises(SystemExit):
        cli.main(['documents:create', 'unused', '--application-id', str(uuid4())])
    database.assert_not_called()


def test_list_text_escapes_names_and_shows_counts(monkeypatch, capsys):
    from app import application
    from app.models.document import DocumentSummary
    row = DocumentSummary(id=uuid4(), name='Report\nSecond\tline', embedding_count=3)
    monkeypatch.setattr(application, 'enumerate_documents', lambda **kwargs: [row])
    cli.main(['documents:list', '--application-id', str(uuid4())])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert 'NAME' in lines[0] and 'EMBEDDINGS' in lines[0]
    assert str(row.id) in lines[1] and 'Report\\nSecond\\tline' in lines[1]
    assert lines[1].endswith('  3')


def test_listing_failure_releases_database_and_prints_no_partial_results(monkeypatch, capsys):
    from unittest.mock import MagicMock
    from app import application
    from app.models.document import DocumentSummary
    database = MagicMock()
    database.__enter__.return_value = database
    database.documents.return_value.list_summaries.side_effect = [
        [DocumentSummary(id=uuid4(), name='name', embedding_count=0)] * 100,
        RuntimeError('Cannot list documents')]
    monkeypatch.setattr(application, 'get_database', lambda: database)
    with pytest.raises(SystemExit):
        cli.main(['documents:list', '--application-id', str(uuid4())])
    assert capsys.readouterr().out == ''
    database.__exit__.assert_called_once()


def test_duplicate_configured_creation_does_not_construct_provider_or_storage(monkeypatch):
    from unittest.mock import MagicMock
    from app import application
    from app.models.application import Application
    database = MagicMock()
    database.__enter__.return_value = database
    database.applications.return_value.get.return_value = Application(name='a', embedding_model='m')
    database.documents.return_value.name_exists.return_value = True
    provider, storage = Mock(), Mock()
    monkeypatch.setattr(application, 'get_database', lambda: database)
    monkeypatch.setattr(application, '_open_configured_provider', provider)
    monkeypatch.setattr('app.storage.factory.get_storage', storage)
    with pytest.raises(ValueError, match='already exists'):
        application.create_document(name='duplicate', path='unused', application_id=uuid4())
    provider.assert_not_called()
    storage.assert_not_called()
    database.__exit__.assert_called_once()


def test_empty_document_list_has_headers(monkeypatch, capsys):
    from app import application
    monkeypatch.setattr(application, 'enumerate_documents', lambda **kwargs: [])
    cli.main(['documents:list', '--application-id', str(uuid4())])
    assert capsys.readouterr().out == f'{"ID":<36}  NAME  STATUS      EMBEDDINGS\n'
