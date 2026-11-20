from types import SimpleNamespace

import pytest
import sqlalchemy as sa

from app import cli
from app.database.backends.postgresql import PostgreSQLBackend
from app.database.postgresql_connection import database_url


@pytest.mark.parametrize('state, expected, guidance', [
    ('42703', 'required column is missing', 'rypiq db:migrate'),
    ('42P01', 'required table is missing', 'rypiq db:migrate'),
    ('42501', 'permission denied', 'configured database user'),
])
def test_document_create_reports_actionable_database_errors(monkeypatch, capsys, state, expected, guidance):
    from app.database.errors import database_errors

    def fail(args, *, config=None):
        driver = Exception('secret password and document contents')
        driver.sqlstate = state
        with database_errors():
            raise sa.exc.ProgrammingError('secret SQL', {'secret': 'value'}, driver)

    monkeypatch.setattr(cli, 'execute', fail)
    with pytest.raises(SystemExit) as caught:
        cli.main(['documents:create', '--name', 'MSCS Program of Study', 'example.pdf',
                  '--application-id', '3fdf113f-00ba-4e16-b9a6-960dd9f8a421'])
    assert caught.value.code == 2
    output = capsys.readouterr()
    assert output.out == ''
    assert expected in output.err
    assert guidance in output.err
    assert 'secret' not in output.err


def test_password_characters_survive_url_construction():
    secret = 'fake%@:secret'
    url = database_url(dict(user='user', password=secret, host='localhost', database='example'))
    assert sa.engine.make_url(url.render_as_string(hide_password=False)).password == secret


def test_driver_errors_are_not_printed_with_credentials(monkeypatch, capsys):
    def fail(args, *, config=None):
        raise sa.exc.OperationalError('secret SQL', {}, Exception('fake%@:secret'))
    monkeypatch.setattr(cli, 'execute', fail)
    monkeypatch.setattr('sys.argv', ['rypiq', 'db:migrate'])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code != 0
    output = capsys.readouterr().err
    assert 'fake%@:secret' not in output
    assert 'secret SQL' not in output
    assert 'pgvector' in output


def test_reset_reports_partial_failure(monkeypatch):
    backend = PostgreSQLBackend(dict(adapter='postgresql', host='localhost', user='tester', database='example'))
    calls = []
    monkeypatch.setattr(backend, 'drop_database', lambda: calls.append('drop') or {'dropped': True})
    def fail():
        raise OSError('failed')
    monkeypatch.setattr(backend, 'create_database', fail)
    try:
        with pytest.raises(RuntimeError, match='data is gone'):
            backend.reset()
        assert calls == ['drop']
    finally:
        backend.close()
