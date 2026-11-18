"""Application listing contracts through the CLI and owned database boundary."""
import json
from unittest.mock import MagicMock, Mock, call
from uuid import UUID

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app import application, cli
from app.models.application import Application


@pytest.fixture
def listing_database(monkeypatch):
    database = MagicMock()
    database.__enter__.return_value = database
    repository = database.applications.return_value
    repository.list.return_value = []
    factory = Mock(return_value=database)
    monkeypatch.setattr(application, 'get_database', factory)
    return database, repository, factory


def records(count):
    return [Application(id=UUID(int=i + 1), name=f'Application {i}',
                        embedding_model='unused', api_key='private-key',
                        data={'private': 'configuration'}) for i in range(count)]


@pytest.mark.parametrize('options', [[], ['--output', 'text']])
def test_text_lists_only_ids_and_names(listing_database, capsys, options):
    database, repository, _ = listing_database
    values = records(2)
    repository.list.return_value = values
    cli.main(['ls-applications', *options])
    output = capsys.readouterr()
    assert output.out == (f'{"ID":<36}  NAME\n'
                          f'{values[0].id}  Application 0\n'
                          f'{values[1].id}  Application 1\n')
    assert output.err == ''
    database.__exit__.assert_called_once_with(None, None, None)


@pytest.mark.parametrize('count', [0, 1, 100, 101, 200])
def test_json_lists_every_page_and_only_public_summary_fields(listing_database, capsys, count):
    database, repository, factory = listing_database
    values = records(count)
    repository.list.side_effect = lambda *, limit, offset: values[offset:offset + limit]
    cli.main(['ls-applications', '--output', 'json'])
    assert json.loads(capsys.readouterr().out) == [
        {'id': str(value.id), 'name': value.name} for value in values]
    assert repository.list.call_args_list == [
        call(limit=100, offset=offset) for offset in range(0, count + 1, 100)]
    factory.assert_called_once_with()
    database.__exit__.assert_called_once_with(None, None, None)
    repository.close.assert_not_called()


def test_empty_text_has_headers(listing_database, capsys):
    cli.main(['ls-applications'])
    assert capsys.readouterr().out == f'{"ID":<36}  NAME\n'


@pytest.mark.parametrize('output', ['text', 'json'])
def test_control_characters_are_escaped_only_in_text(listing_database, capsys, output):
    _, repository, _ = listing_database
    value = records(1)[0]
    value.name = 'Café\nsecond\tcolumn\r\x1b[31m'
    repository.list.return_value = [value]
    cli.main(['ls-applications', '--output', output])
    result = capsys.readouterr().out
    if output == 'json':
        assert json.loads(result)[0]['name'] == value.name
    else:
        assert result == f'{"ID":<36}  NAME\n{value.id}  Café\\nsecond\\tcolumn\\r\\u001b[31m\n'


def test_invalid_output_is_rejected_before_database_access(listing_database, capsys):
    _, _, factory = listing_database
    with pytest.raises(SystemExit) as caught:
        cli.main(['ls-applications', '--output', 'csv'])
    assert caught.value.code == 2
    assert 'invalid choice' in capsys.readouterr().err
    factory.assert_not_called()


@pytest.mark.parametrize('output', ['text', 'json'])
def test_later_page_failure_closes_database_without_partial_output(listing_database, capsys, output):
    database, repository, _ = listing_database
    error = SQLAlchemyError('private connection details')
    repository.list.side_effect = [records(100), error]
    with pytest.raises(SystemExit) as caught:
        cli.main(['ls-applications', '--output', output])
    assert caught.value.code == 2
    result = capsys.readouterr()
    assert result.out == ''
    assert 'Database operation failed' in result.err
    assert 'private connection details' not in result.err
    assert database.__exit__.call_args.args[:2] == (SQLAlchemyError, error)


def test_existing_application_list_keeps_details_and_pagination(listing_database, capsys):
    _, repository, _ = listing_database
    repository.list.return_value = records(1)
    cli.main(['application:list', '--limit', '20', '--offset', '5'])
    result = json.loads(capsys.readouterr().out)
    assert result[0]['embedding_model'] == 'unused'
    assert 'api_key' not in result[0]
    repository.list.assert_called_once_with(limit=20, offset=5)
