from argparse import Namespace
from types import SimpleNamespace

import pytest

from app.database.dispatch import dispatch


@pytest.mark.parametrize('method', ['create', 'migrate', 'drop', 'reset'])
def test_dispatch(method):
    calls = []
    service = SimpleNamespace(**{method: lambda: calls.append(method) or {'ok': True}})
    assert dispatch(Namespace(command=f'db:{method}'), service) == {'ok': True}
    assert calls == [method]


@pytest.mark.parametrize('name', ['upgrade', 'rollback', 'status', 'migration'])
def test_removed_commands(name):
    with pytest.raises(ValueError, match='Unknown database command'):
        dispatch(Namespace(command=f'db:{name}'), object())


def test_python_database_execution_does_not_need_argparse(monkeypatch):
    from unittest.mock import MagicMock
    from app.database import commands
    backend = MagicMock()
    backend.__enter__.return_value = backend
    backend.create_database.return_value = {'created': True}
    monkeypatch.setattr(commands, 'get_database', lambda config=None: backend)
    assert commands.execute_database_command('db:create') == {'created': True}
    backend.create_database.assert_called_once()
    backend.__exit__.assert_called_once()


def test_invalid_python_database_command_does_not_construct_resources(monkeypatch):
    from unittest.mock import Mock
    from app.database import commands
    factory = Mock()
    monkeypatch.setattr(commands, 'get_database', factory)
    with pytest.raises(ValueError, match='Unknown database command'):
        commands.execute_database_command('db:upgrade')
    factory.assert_not_called()


def test_legacy_database_adapter_preserves_result(monkeypatch):
    from app.database import commands
    calls = []
    def execute(name, *, config=None):
        calls.append(name)
        return {'tables_created': []}
    monkeypatch.setattr(commands, 'execute_database_command', execute)
    assert commands.run_database_command(Namespace(command='db:migrate')) == {'tables_created': []}
    assert calls == ['db:migrate']
