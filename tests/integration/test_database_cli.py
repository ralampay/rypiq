import json
import sys
from types import SimpleNamespace
import pytest
from app import cli
from app.database import commands

@pytest.mark.integration
def test_create_delegates_to_configured_backend(monkeypatch, capsys):
    calls = []
    backend = SimpleNamespace(
        create_database=lambda: calls.append("create"),
    )

    monkeypatch.setattr(cli, "get_database", lambda: backend)
    monkeypatch.setattr(sys, "argv", ["rypiq", "db:create"])

    cli.main()

    assert calls == ["create"]
    assert "Database initialized" in capsys.readouterr().out

@pytest.mark.integration
def test_upgrade_dispatches_to_service(monkeypatch):
    calls = []

    monkeypatch.setattr(commands, "upgrade", lambda: calls.append("upgrade"))
    monkeypatch.setattr(sys, "argv", ["rypiq", "db:upgrade"])

    cli.main()

    assert calls == ["upgrade"]
