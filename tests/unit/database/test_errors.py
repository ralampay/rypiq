from types import SimpleNamespace

import pytest
import sqlalchemy as sa

from app import cli
from app.database.backends.postgresql import PostgreSQLBackend
from app.database.postgresql_connection import database_url


def test_password_characters_survive_url_construction():
    secret = 'fake%@:secret'
    url = database_url(dict(user='user', password=secret, host='localhost', database='example'))
    assert sa.engine.make_url(url.render_as_string(hide_password=False)).password == secret


def test_driver_errors_are_not_printed_with_credentials(monkeypatch, capsys):
    def fail(args):
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
