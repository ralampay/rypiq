"""Application listing contracts through the CLI and owned database boundary."""
import json
import re
from unittest.mock import MagicMock, Mock, call
from uuid import UUID

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app import application, cli
from app.models.application import Application, ApplicationSummary


@pytest.fixture
def listing_database(monkeypatch):
    database = MagicMock()
    database.__enter__.return_value = database
    repository = database.applications.return_value
    repository.list_summaries.return_value = []
    factory = Mock(return_value=database)
    monkeypatch.setattr(application, 'get_database', factory)
    return database, repository, factory


def records(count):
    return [ApplicationSummary(id=UUID(int=i + 1), name=f'Application {i}',
                               provision_type='local', embedding_model='unused',
                               embedding_size=768, inference_model=None, document_count=0)
            for i in range(count)]


@pytest.mark.parametrize('options', [[], ['--output', 'text']])
def test_text_lists_model_configuration(listing_database, capsys, options):
    database, repository, _ = listing_database
    values = records(2)
    values[1].document_count = 3
    values[1].provision_type = 'openai'
    values[1].embedding_model = 'text-embedding-model'
    values[1].embedding_size = 1536
    values[1].inference_model = 'chat-model'
    repository.list_summaries.return_value = values
    cli.main(['applications:list', *options])
    output = capsys.readouterr()
    lines = output.out.splitlines()
    assert re.split(r' {2,}', lines[0]) == [
        'ID', 'NAME', 'PROVISION TYPE', 'EMBEDDING MODEL', 'EMBEDDING SIZE', 'INFERENCE MODEL', 'DOCUMENTS']
    assert [re.split(r' {2,}', line) for line in lines[1:]] == [
        [str(values[0].id), 'Application 0', 'local', 'unused', '768', '-', '0'],
        [str(values[1].id), 'Application 1', 'openai', 'text-embedding-model', '1536', 'chat-model', '3']]
    for heading, cells in [('NAME', ['Application 0', 'Application 1']),
                           ('PROVISION TYPE', ['local', 'openai']),
                           ('EMBEDDING MODEL', ['unused', 'text-embedding-model']),
                           ('EMBEDDING SIZE', ['768', '1536']),
                           ('INFERENCE MODEL', ['-', 'chat-model'])]:
        assert all(line.startswith(cell, lines[0].index(heading))
                   for line, cell in zip(lines[1:], cells))
    assert 'private-key' not in output.out
    assert 'configuration' not in output.out
    assert output.err == ''
    database.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize('count', [0, 1, 100, 101, 200])
def test_json_lists_every_page_and_public_model_configuration(listing_database, capsys, count):
    database, repository, factory = listing_database
    values = records(count)
    repository.list_summaries.side_effect = lambda *, limit, offset: values[offset:offset + limit]
    cli.main(['applications:list', '--output', 'json'])
    assert json.loads(capsys.readouterr().out) == [
        {'id': str(value.id), 'name': value.name, 'provision_type': value.provision_type,
         'embedding_model': value.embedding_model, 'embedding_size': value.embedding_size,
         'inference_model': value.inference_model, 'document_count': 0} for value in values]
    assert repository.list_summaries.call_args_list == [
        call(limit=100, offset=offset) for offset in range(0, count + 1, 100)]
    factory.assert_called_once_with()
    database.__exit__.assert_called_once_with(None, None, None)
    repository.close.assert_not_called()


@pytest.mark.parametrize('inference_model', [None, '', 'chat\nmodel'])
def test_json_preserves_model_configuration_types_and_values(listing_database, capsys, inference_model):
    _, repository, _ = listing_database
    value = records(1)[0]
    value.provision_type = 'openai'
    value.embedding_model = 'embedding\tmodel'
    value.embedding_size = 1536
    value.inference_model = inference_model
    value.document_count = 12
    repository.list_summaries.return_value = [value]
    cli.main(['applications:list', '--output', 'json'])
    assert json.loads(capsys.readouterr().out) == [{
        'id': str(value.id), 'name': value.name, 'provision_type': 'openai',
        'embedding_model': 'embedding\tmodel', 'embedding_size': 1536,
        'inference_model': inference_model, 'document_count': 12}]


def test_empty_text_has_headers(listing_database, capsys):
    cli.main(['applications:list'])
    assert capsys.readouterr().out == (
        'ID  NAME  PROVISION TYPE  EMBEDDING MODEL  EMBEDDING SIZE  INFERENCE MODEL  DOCUMENTS\n')


@pytest.mark.parametrize('output', ['text', 'json'])
def test_control_characters_are_escaped_only_in_text(listing_database, capsys, output):
    _, repository, _ = listing_database
    value = records(1)[0]
    value.name = 'Café\nsecond\tcolumn\r\x1b[31m'
    repository.list_summaries.return_value = [value]
    cli.main(['applications:list', '--output', output])
    result = capsys.readouterr().out
    if output == 'json':
        assert json.loads(result)[0]['name'] == value.name
    else:
        assert len(result.splitlines()) == 2
        assert r'Café\nsecond\tcolumn\r\u001b[31m' in result


@pytest.mark.parametrize('model', [None, '', '/models/' + 'long-model-name-' * 12,
                                   'model\nname\tversion\r\x1b[31m'])
def test_text_model_values_are_complete_and_safe(listing_database, capsys, model):
    _, repository, _ = listing_database
    value = records(1)[0]
    value.embedding_model = model
    value.inference_model = model
    repository.list_summaries.return_value = [value]
    cli.main(['applications:list'])
    output = capsys.readouterr().out
    assert len(output.splitlines()) == 2
    cells = re.split(r' {2,}', output.splitlines()[1])
    expected = '-' if not model else json.dumps(model)[1:-1]
    assert cells[3] == expected
    assert cells[5] == expected


def test_invalid_output_is_rejected_before_database_access(listing_database, capsys):
    _, _, factory = listing_database
    with pytest.raises(SystemExit) as caught:
        cli.main(['applications:list', '--output', 'csv'])
    assert caught.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err
    factory.assert_not_called()


@pytest.mark.parametrize('output', ['text', 'json'])
def test_later_page_failure_closes_database_without_partial_output(listing_database, capsys, output):
    database, repository, _ = listing_database
    error = SQLAlchemyError('private connection details')
    repository.list_summaries.side_effect = [records(100), error]
    with pytest.raises(SystemExit) as caught:
        cli.main(['applications:list', '--output', output])
    assert caught.value.code == 2
    result = capsys.readouterr()
    assert result.out == ''
    assert 'Database operation failed' in result.err
    assert 'private connection details' not in result.err
    assert database.__exit__.call_args.args[:2] == (SQLAlchemyError, error)


def test_python_listing_keeps_pagination(listing_database):
    _, repository, _ = listing_database
    values = [Application(name='Existing contract', embedding_model='unused')]
    repository.list.return_value = values
    assert application.enumerate_applications(limit=20, offset=5) == values
    repository.list.assert_called_once_with(limit=20, offset=5)


@pytest.mark.parametrize('command', ['ls-applications', 'application:list',
                         'application:create', 'application:show', 'application:update',
                         'application:delete'])
def test_old_command_names_are_rejected(listing_database, capsys, command):
    _, _, factory = listing_database
    with pytest.raises(SystemExit) as caught:
        cli.main([command])
    assert caught.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err
    factory.assert_not_called()


@pytest.mark.parametrize('deleted', [True, False])
def test_delete_returns_repository_outcome_and_closes_database(listing_database, capsys, deleted):
    database, repository, _ = listing_database
    repository.delete.return_value = deleted
    application_id = UUID(int=1)
    cli.main(['applications:delete', str(application_id)])
    assert json.loads(capsys.readouterr().out) == {'deleted': deleted}
    repository.delete.assert_called_once_with(application_id)
    database.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize('error, message', [
    (ValueError('Cannot delete application while it has documents'), 'Cannot delete application while it has documents'),
    (SQLAlchemyError('private driver detail'), 'Database operation failed'),
])
def test_delete_failure_is_safe_and_closes_database(listing_database, capsys, error, message):
    database, repository, _ = listing_database
    repository.delete.side_effect = error
    with pytest.raises(SystemExit) as caught:
        cli.main(['applications:delete', str(UUID(int=1))])
    assert caught.value.code == 2
    result = capsys.readouterr()
    assert result.out == ''
    assert message in result.err
    assert 'private driver detail' not in result.err
    assert database.__exit__.call_args.args[:2] == (type(error), error)


def test_delete_invalid_uuid_never_opens_database(listing_database, capsys):
    _, _, factory = listing_database
    with pytest.raises(SystemExit) as caught:
        cli.main(['applications:delete', 'invalid'])
    assert caught.value.code == 2
    assert 'invalid UUID' in capsys.readouterr().err
    factory.assert_not_called()


def test_create_show_and_name_update_keep_contracts(listing_database, capsys):
    database, repository, _ = listing_database
    database.settings = {'dimensions': 768}
    repository.create.side_effect = lambda value: value
    cli.main(['applications:create', '--name', 'Original', '--embedding-model', 'model'])
    created = json.loads(capsys.readouterr().out)
    value = repository.create.call_args.args[0]
    assert created['api_key'] == value.api_key
    assert created['name'] == 'Original'
    repository.get.return_value = value
    cli.main(['applications:show', str(value.id)])
    shown = json.loads(capsys.readouterr().out)
    assert shown == {key: item for key, item in created.items() if key != 'api_key'}
    repository.update.side_effect = lambda updated: updated
    cli.main(['applications:update', str(value.id), '--name', 'Renamed'])
    updated = json.loads(capsys.readouterr().out)
    assert updated == dict(shown, name='Renamed')
    assert repository.update.call_args.args[0].api_key == value.api_key
