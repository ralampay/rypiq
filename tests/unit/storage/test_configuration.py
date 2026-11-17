import pytest
import yaml

from app.storage.config import load_storage_config


def configure(tmp_path, monkeypatch, body):
    path = tmp_path / 'config/storage.yml'
    path.parent.mkdir()
    path.write_text(yaml.safe_dump(body))
    monkeypatch.setenv('RYPIQ_STORAGE_CONFIG', str(path))
    monkeypatch.setenv('DATABASE_ENV', 'test')
    monkeypatch.delenv('STORAGE_ENV', raising=False)


def test_relative_storage_root_is_independent_of_working_directory(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch, {'test': {'adapter': 'local', 'root': 'objects'}})
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert load_storage_config()['root'] == str(tmp_path / 'objects')


@pytest.mark.parametrize('body', [[], {'other': {}}])
def test_bad_storage_configuration(tmp_path, monkeypatch, body):
    configure(tmp_path, monkeypatch, body)
    with pytest.raises(ValueError):
        load_storage_config()


def test_adapter_selection_stays_in_factory(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch, {'test': {'adapter': 'unsupported'}})
    assert load_storage_config()['adapter'] == 'unsupported'
