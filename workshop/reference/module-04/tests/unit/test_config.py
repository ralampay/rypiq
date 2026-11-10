from pathlib import Path
import pytest
from ragworkshop.config import load_settings
from ragworkshop.storage.factory import get_storage


def test_precedence_and_dotenv(monkeypatch, tmp_path):
    dotenv = tmp_path / "settings.env"
    dotenv.write_text("RAG_WORKSHOP_ENV=production\nRAG_WORKSHOP_STORAGE_ROOT=/unused\n")
    monkeypatch.setenv("RAG_WORKSHOP_DOTENV", str(dotenv))
    selected = tmp_path / "selected"
    settings = load_settings({"storage_root": selected})
    assert settings.environment == "test"
    assert settings.storage_root == selected
    assert settings.database_path == tmp_path / "db"


def test_yaml_relative_paths_and_isolation(monkeypatch, tmp_path):
    monkeypatch.delenv("RAG_WORKSHOP_DATABASE_PATH")
    monkeypatch.delenv("RAG_WORKSHOP_STORAGE_ROOT")
    test = load_settings()
    dev = load_settings({"env": "development"})
    assert test.database_path != dev.database_path
    assert test.storage_root != dev.storage_root
    assert test.storage_root == (Path.cwd() / ".data/test/objects").resolve()
    # Reading settings creates neither environment's persistence directories.


def test_invalid_config(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="Unknown"):
        load_settings({"env": "typo"})
    bad = tmp_path / "bad.yml"
    bad.write_text("[]")
    with pytest.raises(ValueError, match="section"):
        load_settings({"storage_config": bad})
    with pytest.raises(ValueError, match="Unsupported"):
        get_storage("s3", tmp_path)
