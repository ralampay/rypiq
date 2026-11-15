from argparse import Namespace
from types import SimpleNamespace

import pytest

from app.database.dispatch import dispatch

@pytest.mark.parametrize(
    ("command", "method", "arguments"),
    [
        ("db:create", "create", ()),
        ("db:drop", "drop", ()),
        ("db:upgrade", "generate", ()),
        ("db:migrate", "migrate", ()),
        ("db:rollback", "rollback", ()),
        ("db:status", "status", ()),
        ("db:migration", "new_migration", ("add_index",)),
    ],
)
def test_each_command_has_one_purpose(command, method, arguments):
    calls = []
    services = {}
    
    for name in (
        "create", "drop", "generate", "migrate", "rollback", "status", "new_migration",
    ):
        def record(*args, name=name):
            calls.append((name, args))

            return {
                "method": name 
            }

        services[name] = record

    result = dispatch(
        Namespace(command=command, name="add_index"),
        SimpleNamespace(**services),
    )

    assert calls == [(method, arguments)]
    assert result == { 
        "method": method,
    }

def test_unknown_command_does_not_call_a_service():
    with pytest.raises(ValueError, match="Unknown database command db:unknown"):
        dispatch(Namespace(command="db:unknown"), object())
