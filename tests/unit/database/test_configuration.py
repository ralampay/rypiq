from dataclasses import fields

import pytest
import yaml

from app.database.config import load_database_config
from app.models.embedding import Embedding


def configure(tmp_path, monkeypatch, **overrides):
    settings = dict(adapter='postgresql', host='localhost', database='rypiq_test', user='tester')
    settings.update(overrides)
    path = tmp_path / 'database.yml'
    path.write_text(yaml.safe_dump({'test': settings}))
    monkeypatch.setenv('RYPIQ_DATABASE_CONFIG', str(path))
    monkeypatch.setenv('DATABASE_ENV', 'test')


def test_defaults_and_password_environment(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch, password_env='RYPIQ_TEST_SECRET')
    monkeypatch.setenv('RYPIQ_TEST_SECRET', 'test%@:password')
    config = load_database_config()
    assert config['dimensions'] == 768
    assert config['port'] == 5432
    assert config['password'] == 'test%@:password'


@pytest.mark.parametrize('dimension', [0, -1, True, 1.5, '3', 16001])
def test_rejects_invalid_dimensions(tmp_path, monkeypatch, dimension):
    configure(tmp_path, monkeypatch, dimensions=dimension)
    with pytest.raises(ValueError, match='dimensions'):
        load_database_config()


def test_configurable_dimensions(tmp_path, monkeypatch):
    configure(tmp_path, monkeypatch, dimensions=3)
    assert load_database_config()['dimensions'] == 3


def test_embedding_has_model_name_without_discovery_metadata():
    names = {field.name for field in fields(Embedding)}
    assert 'model_name' in names
    assert 'embedding_model' in names
    assert not hasattr(Embedding, '__persistence__')


@pytest.mark.parametrize('adapter', ['chromadb', 'pgvector', None])
def test_only_postgresql_is_an_adapter(tmp_path, monkeypatch, adapter):
    configure(tmp_path, monkeypatch, adapter=adapter, path='data')
    with pytest.raises(ValueError, match='adapter'):
        load_database_config()


@pytest.mark.parametrize('database', ['postgres', 'template0', 'template1', 'a' * 64, '界' * 22])
def test_rejects_system_or_truncated_database_names(tmp_path, monkeypatch, database):
    configure(tmp_path, monkeypatch, database=database)
    with pytest.raises(ValueError):
        load_database_config()


def test_bad_yaml_does_not_expose_contents(tmp_path, monkeypatch):
    path = tmp_path / 'database.yml'
    path.write_text('test: [password: fake-secret, invalid')
    monkeypatch.setenv('RYPIQ_DATABASE_CONFIG', str(path))
    with pytest.raises(ValueError) as error:
        load_database_config()
    assert 'fake-secret' not in str(error.value)
