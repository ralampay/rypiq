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
