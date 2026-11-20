import importlib
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import create_app
from app.exceptions import ConfigurationError, DatabaseConfigurationError


def config_module():
    import app
    assert (Path(app.__file__).parent / 'config.py').exists(), 'Shared configuration module is required'
    return importlib.import_module('app.config')


def test_explicit_api_config_works_without_environment(monkeypatch, tmp_path):
    module = config_module()
    from app.logging.config import LoggingSettings
    monkeypatch.setenv('APP_SECRET_KEY', 'ambient-must-not-win')
    settings = module.AppConfig(api=module.ApiSettings(secret_key='injected'),
                                logging=LoggingSettings(directory=tmp_path / 'logs'))
    with TestClient(create_app(config=settings)) as client:
        assert client.app.state.config is settings
        assert client.get('/api/v1/health', headers={'X-App-Secret-Key': 'injected'}).status_code == 200
        assert client.get('/api/v1/health', headers={'X-App-Secret-Key': 'ambient-must-not-win'}).status_code == 401
    assert 'injected' not in repr(settings)


def test_load_captures_files_credentials_and_environment(tmp_path, monkeypatch):
    module = config_module()
    directory = tmp_path / 'config'
    directory.mkdir()
    (tmp_path / '.env').write_text('APP_SECRET_KEY=dotenv-key\nDB_PASSWORD=dotenv-password\n')
    database = directory / 'database.yml'
    database.write_text('development:\n  adapter: postgresql\n  host: first\n  user: test\n  database: example\n  password_env: DB_PASSWORD\n')
    (directory / 'storage.yml').write_text('development:\n  adapter: local\n  root: objects\n')
    (directory / 'models.yml').write_text('models: {}\n')
    environment = {'APP_SECRET_KEY': 'process-key', 'DB_PASSWORD': 'process-password', 'CUSTOM_KEY': 'captured'}
    snapshot = module.load_config(environ=environment, project_root=tmp_path)
    environment['CUSTOM_KEY'] = 'changed'
    database.write_text(database.read_text().replace('first', 'second'))
    monkeypatch.setenv('DB_PASSWORD', 'ambient-password')
    assert snapshot.require_api().secret_key == 'process-key'
    assert snapshot.require_database().host == 'first'
    assert snapshot.require_database().password == 'process-password'
    assert snapshot.require_storage().root == tmp_path / 'objects'
    assert snapshot.credentials['CUSTOM_KEY'] == 'captured'
    assert module.load_config(environ=environment, project_root=tmp_path).require_database().host == 'second'
    assert 'process-password' not in repr(snapshot.require_database())
    with pytest.raises(TypeError):
        snapshot.credentials['CUSTOM_KEY'] = 'mutated'


def test_missing_sections_are_deferred_and_replaceable(tmp_path):
    module = config_module()
    config = module.load_config(environ={}, project_root=tmp_path)
    with pytest.raises(DatabaseConfigurationError):
        config.require_database()
    with pytest.raises(ConfigurationError, match='APP_SECRET_KEY'):
        config.require_api()
    changed = replace(config, database=module.DatabaseSettings(host='localhost', user='test', database='example'))
    assert changed.require_database().dimensions == 768
    assert config.database is None


def test_explicit_config_does_not_read_environment_or_files(monkeypatch, tmp_path):
    import app.api
    module = config_module()
    config = module.AppConfig(api=module.ApiSettings(secret_key='explicit'))
    monkeypatch.setattr(app.api, 'load_config', lambda **_: pytest.fail('Unexpected configuration reload'))
    monkeypatch.setattr(Path, 'read_text', lambda *_args, **_kwargs: pytest.fail('Unexpected file read'))
    api = create_app(config=config)
    assert api.state.config is config


def test_two_api_instances_keep_auth_cors_storage_and_database_isolated(monkeypatch, tmp_path):
    from unittest.mock import MagicMock
    from uuid import uuid4
    from app import application
    from app.config import AppConfig, ApiSettings, DatabaseSettings, StorageSettings
    from app.models.application import Application
    from app.logging.config import LoggingSettings
    first = AppConfig(api=ApiSettings(secret_key='first', cors_origins=('https://first.example',)),
                      database=DatabaseSettings(host='first-host', user='test', database='first-db'),
                      storage=StorageSettings(tmp_path / 'first'), logging=LoggingSettings(directory=tmp_path / 'logs'))
    second = replace(first, api=ApiSettings(secret_key='second', cors_origins=('https://second.example',)),
                     database=replace(first.database, host='second-host', database='second-db'),
                     storage=StorageSettings(tmp_path / 'second'))
    seen = []
    def database(config):
        seen.append(config)
        backend = MagicMock()
        backend.__enter__.return_value = backend
        backend.applications.return_value.get.return_value = Application(
            name=config.require_database().database, embedding_model='fake')
        return backend
    monkeypatch.setattr(application, 'get_database', database)
    with TestClient(create_app(config=first)) as one, TestClient(create_app(config=second)) as two:
        monkeypatch.setenv('APP_SECRET_KEY', 'changed')
        for client, config, other in ((one, first, second), (two, second, first)):
            headers = {'X-App-Secret-Key': config.api.secret_key, 'Origin': config.api.cors_origins[0]}
            response = client.get(f'/api/v1/applications/{uuid4()}', headers=headers)
            assert response.json()['name'] == config.database.database
            assert response.headers['access-control-allow-origin'] == config.api.cors_origins[0]
            assert client.get('/api/v1/health', headers={'X-App-Secret-Key': other.api.secret_key}).status_code == 401
            response = client.post('/api/v1/storage', headers=headers, files={'file': ('file.txt', b'example')})
            assert response.status_code == 201
            assert (config.storage.root / response.json()['key']).exists()
            assert not (other.storage.root / response.json()['key']).exists()
    assert seen == [first, second]


def test_provider_credentials_are_explicit_and_do_not_fall_back(monkeypatch):
    from unittest.mock import Mock
    from app.config import AppConfig
    from app.providers.config import ProviderSettings
    from app.providers.factory import open_capability_provider
    from app.exceptions import ProviderConfigurationError
    settings = ProviderSettings(provider='openai', capability='embedding', model='fake', embedding_size=3,
                                api_key_env='CUSTOM_KEY')
    values = {'CUSTOM_KEY': 'captured-key'}
    config = AppConfig(credentials=values)
    values['CUSTOM_KEY'] = 'changed'
    monkeypatch.setenv('CUSTOM_KEY', 'ambient-key')
    client = Mock()
    factory = Mock(return_value=client)
    with open_capability_provider(settings, credentials=config.credentials, client_factory=factory):
        assert factory.call_args.kwargs['api_key'] == 'captured-key'
    client.close.assert_called_once()
    with pytest.raises(ProviderConfigurationError):
        with open_capability_provider(settings, credentials={}, client_factory=factory):
            pytest.fail('Missing injected credentials must not use process values')
    assert 'captured-key' not in repr(config.credentials)


def test_catalog_is_captured_and_defensive_copies_are_independent(tmp_path):
    from app.config import load_config
    path = tmp_path / 'config'
    path.mkdir()
    catalog = path / 'models.yml'
    catalog.write_text('models:\n  embed:\n    provider: local\n    capabilities: [embedding]\n    path: old.gguf\n    options: {context_size: 512}\n')
    config = load_config(environ={}, project_root=tmp_path)
    catalog.unlink()
    first = config.require_catalog()
    first['embed'].options['context_size'] = 42
    first['embed'] = replace(first['embed'], model='different')
    assert config.require_catalog()['embed'].options['context_size'] == 512
    assert config.require_catalog()['embed'].model == str(path / 'old.gguf')
    with pytest.raises(TypeError):
        config.catalog['embed'].options['context_size'] = 1


def test_doctor_checks_snapshot_but_performs_fresh_readiness(monkeypatch, tmp_path):
    from unittest.mock import Mock
    from app.config import AppConfig, DatabaseSettings, StorageSettings
    from app import diagnostics
    config = AppConfig(database=DatabaseSettings(host='captured', user='test', database='example'),
                       storage=StorageSettings(tmp_path / 'objects'), catalog={})
    check = Mock(return_value=[])
    monkeypatch.setattr(diagnostics, 'check_database', check)
    monkeypatch.setattr(diagnostics, 'load_config', lambda: pytest.fail('No reload'))
    monkeypatch.setattr(Path, 'read_text', lambda *_args, **_kwargs: pytest.fail('No configuration read'))
    for _ in range(2):
        assert diagnostics.check_configuration(config=config).incomplete_count == 0
    assert check.call_count == 2
    assert check.call_args.kwargs['settings']['host'] == 'captured'
    assert not config.storage.root.exists()


def test_cli_injected_text_does_not_require_database_or_api_secret(monkeypatch, tmp_path, capsys):
    import json
    from app import cli
    from app.config import AppConfig
    from app.logging.config import LoggingSettings
    config = AppConfig(logging=LoggingSettings(directory=tmp_path / 'logs'))
    monkeypatch.setattr(cli, 'load_config', lambda: pytest.fail('No environment reload'))
    cli.main(['process', 'text', 'hello world'], config=config)
    assert json.loads(capsys.readouterr().out)['words'] == 2


def test_worker_loads_once_and_reuses_config_between_polls(monkeypatch):
    from app import worker
    from app.config import AppConfig
    from unittest.mock import Mock
    config = AppConfig()
    loader = Mock(return_value=config)
    monkeypatch.setattr(worker, 'load_config', loader)
    callbacks = {}
    monkeypatch.setattr(worker.signal, 'signal', lambda signum, callback: callbacks.setdefault(signum, callback))
    seen = []
    def run_once(*, config):
        seen.append(config)
        if len(seen) == 2:
            callbacks[worker.signal.SIGTERM]()
    monkeypatch.setattr(worker, 'run_once', run_once)
    worker.run_worker()
    loader.assert_called_once_with()
    assert seen == [config, config]


def test_typed_factories_preserve_credentials_and_cleanup(monkeypatch, tmp_path):
    from app.config import AppConfig, DatabaseSettings, StorageSettings
    from app.database.factory import get_database
    from app.storage.factory import get_storage
    config = AppConfig(database=DatabaseSettings(host='localhost', user='test', database='example', password='injected'),
                       storage=StorageSettings(tmp_path / 'objects'))
    monkeypatch.setenv('DATABASE_ENV', 'missing')
    with get_database(config) as database:
        assert database.settings['password'] == 'injected'
    storage = get_storage(config)
    assert storage.list() == []


@pytest.mark.parametrize('section,contents', [
    ('database', 'development: [invalid'), ('storage', 'development: [invalid'),
    ('logging', 'development: [invalid'), ('models', 'models: [invalid'),
])
def test_invalid_optional_sections_do_not_block_api_health(tmp_path, section, contents):
    from app.config import load_config
    from app.logging.config import LoggingSettings
    directory = tmp_path / 'config'
    directory.mkdir()
    (directory / (section + '.yml')).write_text(contents)
    config = load_config(environ={'APP_SECRET_KEY': 'key'}, project_root=tmp_path)
    # Logging is a process startup capability, so supply it explicitly for this API.
    config = replace(config, logging=LoggingSettings(directory=tmp_path / 'logs'))
    with TestClient(create_app(config=config)) as client:
        assert client.get('/api/v1/health', headers={'X-App-Secret-Key': 'key'}).status_code == 200
    accessor = 'require_catalog' if section == 'models' else 'require_' + section
    if section != 'logging':
        with pytest.raises(ConfigurationError):
            getattr(config, accessor)()


def test_direct_logging_paths_are_fixed_when_snapshot_is_created(monkeypatch, tmp_path):
    from app.config import AppConfig
    from app.logging.config import LoggingSettings
    monkeypatch.setenv('HOME', str(tmp_path / 'first'))
    config = AppConfig(logging=LoggingSettings(directory=Path('~/logs')))
    monkeypatch.setenv('HOME', str(tmp_path / 'second'))
    assert config.require_logging().directory == tmp_path / 'first' / 'logs'


def test_cli_injected_server_reload_has_safe_actionable_error(capsys):
    from app import cli
    from app.config import AppConfig, ApiSettings
    with pytest.raises(SystemExit):
        cli.main(['server', '--reload'], config=AppConfig(api=ApiSettings(secret_key='key')))
    assert 'Injected server configuration cannot be combined with --reload' in capsys.readouterr().err


def test_empty_injected_storage_root_cannot_target_working_directory():
    from app.config import StorageSettings
    from app.exceptions import StorageConfigurationError
    with pytest.raises(StorageConfigurationError, match='nonempty'):
        StorageSettings(root='')
