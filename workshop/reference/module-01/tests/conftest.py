import pytest


def pytest_addoption(parser):
    parser.addoption("--run-real-model", action="store_true", default=False)


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-real-model"):
        for item in items:
            if "real_model" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="requires --run-real-model"))


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch, tmp_path):
    import os
    for key in list(os.environ):
        if key.startswith("RAG_WORKSHOP_") and not key.startswith("RAG_WORKSHOP_TEST_"):
            monkeypatch.delenv(key)
    monkeypatch.setenv("RAG_WORKSHOP_ENV", "test")
    monkeypatch.setenv("RAG_WORKSHOP_DATABASE_PATH", str(tmp_path / "db"))
    monkeypatch.setenv("RAG_WORKSHOP_STORAGE_ROOT", str(tmp_path / "objects"))
    monkeypatch.setenv("RAG_WORKSHOP_DOTENV", str(tmp_path / "absent.env"))
