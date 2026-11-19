"""Environment preload behavior through public configuration boundaries."""
import os
from unittest.mock import Mock

import pytest
import yaml

from app import cli, diagnostics, environment
from app.database.config import load_database_config
from app.exceptions import ConfigurationError
from app.providers.catalog import load_catalog
from app.storage.config import load_storage_config


@pytest.fixture
def local_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(environment, 'PROJECT_ROOT', tmp_path)
    # Restore the entire environment, including keys added by python-dotenv.
    monkeypatch.setattr(os, 'environ', os.environ.copy())
    for name in ('DATABASE_ENV', 'STORAGE_ENV', 'RYPIQ_DATABASE_CONFIG',
                 'RYPIQ_STORAGE_CONFIG', 'RYPIQ_MODELS_CONFIG', 'PRELOAD_TEST_PASSWORD',
                 'PYTHON_DOTENV_DISABLED'):
        monkeypatch.delenv(name, raising=False)
    database = tmp_path / 'database.yml'
    database.write_text(yaml.safe_dump({'testing': {
        'adapter': 'postgresql', 'host': 'localhost', 'user': 'tester',
        'database': 'example', 'password_env': 'PRELOAD_TEST_PASSWORD'}}))
    storage = tmp_path / 'storage.yml'
    storage.write_text(yaml.safe_dump({'testing': {'adapter': 'local', 'root': str(tmp_path)}}))
    catalog = tmp_path / 'models.yml'
    catalog.write_text('models: {}\n')
    dotenv = tmp_path / '.env'
    dotenv.write_text(f'DATABASE_ENV=testing\nRYPIQ_DATABASE_CONFIG={database}\n'
                      f'RYPIQ_STORAGE_CONFIG={storage}\nRYPIQ_MODELS_CONFIG={catalog}\n'
                      'PRELOAD_TEST_PASSWORD=local-secret\n')
    return dotenv


def test_database_loads_environment_before_selecting_yaml(local_environment):
    assert load_database_config()['password'] == 'local-secret'


@pytest.mark.parametrize('value', ['exported-secret', ''])
def test_exported_values_win(local_environment, monkeypatch, value):
    monkeypatch.setenv('PRELOAD_TEST_PASSWORD', value)
    assert load_database_config()['password'] == value


def test_storage_loads_environment_before_selecting_yaml(local_environment):
    assert load_storage_config()['root'] == str(local_environment.parent)


def test_catalog_loads_environment_before_selecting_yaml(local_environment):
    assert load_catalog() == {}


def test_diagnostics_preloads_environment(local_environment, monkeypatch):
    check = Mock(return_value=[])
    monkeypatch.setattr(diagnostics, 'check_database', check)
    report = diagnostics.check_configuration()
    assert report.environment == 'testing'
    assert report.incomplete_count == 0
    assert check.call_args.kwargs['settings']['password'] == 'local-secret'


def test_cli_preloads_before_dispatch(local_environment, monkeypatch, capsys):
    def execute(args):
        assert os.environ['PRELOAD_TEST_PASSWORD'] == 'local-secret'
        return {}
    monkeypatch.setattr(cli, 'execute', execute)
    cli.main(['applications:list'])
    assert 'local-secret' not in capsys.readouterr().out


def test_missing_file_is_optional(local_environment):
    local_environment.unlink()
    environment.load_environment()
    assert 'PRELOAD_TEST_PASSWORD' not in os.environ


def test_missing_password_keeps_existing_error(local_environment):
    local_environment.write_text(local_environment.read_text().replace('PRELOAD_TEST_PASSWORD=local-secret', ''))
    with pytest.raises(ConfigurationError, match='environment variable is missing'):
        load_database_config()


@pytest.mark.parametrize('failure', ['encoding', 'permission'])
def test_read_failures_are_safe(local_environment, monkeypatch, failure):
    if failure == 'encoding':
        local_environment.write_bytes(b'PRIVATE_SECRET=\xff')
    else:
        monkeypatch.setattr(type(local_environment), 'open', Mock(side_effect=PermissionError('PRIVATE_SECRET')))
    with pytest.raises(ConfigurationError) as error:
        environment.load_environment()
    assert 'PRIVATE_SECRET' not in str(error.value)


def test_cli_read_failure_prevents_dispatch(local_environment, monkeypatch, capsys):
    local_environment.write_bytes(b'PRIVATE_SECRET=\xff')
    execute = Mock()
    monkeypatch.setattr(cli, 'execute', execute)
    with pytest.raises(SystemExit) as error:
        cli.main(['applications:list'])
    assert error.value.code != 0
    execute.assert_not_called()
    assert 'PRIVATE_SECRET' not in capsys.readouterr().err


def test_repeated_load_preserves_process_values(local_environment):
    environment.load_environment()
    local_environment.write_text('PRELOAD_TEST_PASSWORD=changed\n')
    environment.load_environment()
    assert os.environ['PRELOAD_TEST_PASSWORD'] == 'local-secret'
