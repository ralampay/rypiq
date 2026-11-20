from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from app import cli, application
from app import diagnostics


@pytest.fixture
def configuration(tmp_path, monkeypatch):
    database = tmp_path / 'config/database.yml'
    storage = tmp_path / 'config/storage.yml'
    models = tmp_path / 'config/models.yml'
    database.parent.mkdir()
    database.write_text(yaml.safe_dump({'development': {
        'adapter': 'postgresql', 'host': 'localhost', 'user': 'example',
        'database': 'example', 'password_env': 'DOCTOR_TEST_PASSWORD'}}))
    storage.write_text(yaml.safe_dump({'development': {'adapter': 'local', 'root': 'objects/new'}}))
    monkeypatch.setenv('DATABASE_ENV', 'development')
    monkeypatch.delenv('STORAGE_ENV', raising=False)
    monkeypatch.setenv('RYPIQ_DATABASE_CONFIG', str(database))
    monkeypatch.setenv('RYPIQ_STORAGE_CONFIG', str(storage))
    monkeypatch.setenv('RYPIQ_MODELS_CONFIG', str(models))
    monkeypatch.setenv('DOCTOR_TEST_PASSWORD', 'password-value-never-print')
    monkeypatch.setattr(application, 'get_database', Mock(side_effect=AssertionError('Database I/O forbidden')))
    monkeypatch.setattr(diagnostics, 'check_database', Mock(return_value=[]), raising=False)
    return database, storage, models


def invoke(capsys):
    try:
        cli.main(['--doctor'])
        code = 0
    except SystemExit as error:
        code = error.code
    captured = capsys.readouterr()
    assert captured.err == ''
    assert 'password-value-never-print' not in captured.out
    return code, captured.out


def test_ready_configuration_does_not_create_storage_or_application_resources(configuration, capsys):
    database, storage, models = configuration
    before = set(database.parent.parent.rglob('*'))
    code, report = invoke(capsys)
    assert code == 0
    assert 'COMPLETE' in report and 'OPTIONAL' in report
    assert 'development' in report and str(database) in report and str(storage) in report
    assert str(models) in report
    assert 'Saved applications were not checked' in report
    assert set(database.parent.parent.rglob('*')) == before
    application.get_database.assert_not_called()


def test_missing_password_has_action_and_other_checks_continue(configuration, monkeypatch, capsys):
    monkeypatch.delenv('DOCTOR_TEST_PASSWORD')
    code, report = invoke(capsys)
    assert code == 1
    assert 'REQUIRED' in report and 'DOCTOR_TEST_PASSWORD' in report
    assert 'export' in report.lower()
    assert 'Storage' in report and 'Model catalog' in report


@pytest.mark.parametrize('contents', ['[private-secret', '[]', 'development: null', 'other: {}',
                                     'development: {adapter: sqlite}',
                                     'development: {adapter: postgresql, host: localhost, user: example, database: postgres}'])
def test_invalid_database_configuration_is_safe(configuration, capsys, contents):
    configuration[0].write_text(contents)
    code, report = invoke(capsys)
    assert code == 1
    assert 'private-secret' not in report
    assert 'Storage' in report
    assert 'INVALID' in report or 'REQUIRED' in report


def test_missing_database_fields_are_named(configuration, capsys):
    configuration[0].write_text('development: {adapter: postgresql}')
    code, report = invoke(capsys)
    assert code == 1
    assert all(key in report for key in ('host', 'user', 'database'))


def test_password_variable_not_required_when_unreferenced(configuration, monkeypatch, capsys):
    path = configuration[0]
    data = yaml.safe_load(path.read_text())
    del data['development']['password_env']
    path.write_text(yaml.safe_dump(data))
    monkeypatch.delenv('DOCTOR_TEST_PASSWORD')
    assert invoke(capsys)[0] == 0


@pytest.mark.parametrize('index', [0, 1])
def test_missing_required_file(configuration, capsys, index):
    configuration[index].unlink()
    code, report = invoke(capsys)
    assert code == 1 and 'REQUIRED' in report
    assert str(configuration[index]) in report


@pytest.mark.parametrize('contents', ['development: {adapter: s3, root: objects}',
                                     'development: {adapter: local}',
                                     'development: {adapter: local, root: 123}',
                                     'development: []', 'development: [broken-private-secret'])
def test_invalid_storage_configuration(configuration, capsys, contents):
    configuration[1].write_text(contents)
    code, report = invoke(capsys)
    assert code == 1
    assert 'broken-private-secret' not in report
    assert 'Storage' in report and 'Model catalog' in report


def test_storage_environment_override(configuration, monkeypatch, capsys):
    monkeypatch.setenv('STORAGE_ENV', 'production')
    code, report = invoke(capsys)
    assert code == 1 and 'production' in report


def test_storage_root_is_file(configuration, capsys):
    _, path, _ = configuration
    path.write_text(yaml.safe_dump({'development': {'adapter': 'local', 'root': str(path)}}))
    code, report = invoke(capsys)
    assert code == 1 and 'directory' in report.lower()


def test_inaccessible_storage_root(configuration, monkeypatch, capsys):
    import os
    real_access = os.access
    root = configuration[0].parent.parent
    monkeypatch.setattr(os, 'access', lambda path, mode: False if Path(path) == root else real_access(path, mode))
    code, report = invoke(capsys)
    assert code == 1 and 'permission' in report.lower()


def test_optional_catalog_reports_model_and_credentials_without_loading(configuration, monkeypatch, capsys):
    configuration[2].write_text(yaml.safe_dump({'models': {
        'local': {'provider': 'local', 'capabilities': ['embedding'], 'path': 'missing.gguf'},
        'remote': {'provider': 'openai', 'capabilities': ['embedding'], 'model': 'remote-model',
                   'options': {'api_key_env': 'DOCTOR_TEST_API_KEY'}}}}))
    monkeypatch.delenv('DOCTOR_TEST_API_KEY', raising=False)
    code, report = invoke(capsys)
    assert code == 0
    assert 'missing.gguf' in report and 'DOCTOR_TEST_API_KEY' in report
    assert 'OPTIONAL' in report


def test_valid_catalog_model_and_credentials(configuration, monkeypatch, capsys):
    model = configuration[2].parent / 'model.gguf'
    model.write_bytes(b'not a model: must never load this')
    configuration[2].write_text(yaml.safe_dump({'models': {
        'local': {'provider': 'local', 'capabilities': ['embedding'], 'path': 'model.gguf'},
        'remote': {'provider': 'openai', 'capabilities': ['embedding'], 'model': 'remote-model',
                   'options': {'api_key_env': 'DOCTOR_TEST_API_KEY'}}}}))
    monkeypatch.setenv('DOCTOR_TEST_API_KEY', 'api-value-never-print')
    code, report = invoke(capsys)
    assert code == 0 and 'api-value-never-print' not in report
    assert 'COMPLETE' in report and 'model.gguf' in report


def test_invalid_catalog_is_optional(configuration, capsys):
    configuration[2].write_text('models: [private-secret')
    code, report = invoke(capsys)
    assert code == 0 and 'OPTIONAL' in report
    assert 'private-secret' not in report


@pytest.mark.parametrize('args', [[], ['--doctor', 'applications:list'], ['applications:list', '--doctor']])
def test_doctor_cannot_be_combined_with_commands_and_command_is_otherwise_required(args, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(args)
    assert exc.value.code == 2
    assert capsys.readouterr().out == ''


def test_openai_catalog_checks_default_credential_variable(configuration, monkeypatch, capsys):
    configuration[2].write_text(yaml.safe_dump({'models': {
        'remote': {'provider': 'openai', 'capabilities': ['embedding'], 'model': 'remote-model'}}}))
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    code, report = invoke(capsys)
    assert code == 0 and 'OPENAI_API_KEY' in report


def test_existing_storage_directory_is_complete(configuration, capsys):
    directory = configuration[0].parent.parent / 'objects/new'
    directory.mkdir(parents=True)
    code, report = invoke(capsys)
    assert code == 0 and 'directory permissions available' in report


def test_unreadable_configuration_does_not_expose_os_error(configuration, monkeypatch, capsys):
    original = Path.open
    def read(path, *args, **kwargs):
        if path == configuration[0]:
            raise PermissionError('private-os-detail')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', read)
    code, report = invoke(capsys)
    assert code == 1 and 'permissions' in report
    assert 'private-os-detail' not in report
    assert 'Storage' in report


def test_invalid_paths_do_not_stop_other_checks(configuration, monkeypatch, capsys):
    monkeypatch.setenv('RYPIQ_DATABASE_CONFIG', '~nonexistent_doctor_test_user/config.yml')
    code, report = invoke(capsys)
    assert code == 1 and 'Storage' in report and 'Model catalog' in report


def test_invalid_port_is_reported(configuration, capsys):
    path = configuration[0]
    data = yaml.safe_load(path.read_text())
    data['development']['port'] = 0
    path.write_text(yaml.safe_dump(data))
    code, report = invoke(capsys)
    assert code == 1 and 'port' in report


def test_control_characters_in_environment_cannot_break_report_rows(configuration, monkeypatch, capsys):
    monkeypatch.setenv('DATABASE_ENV', 'missing\nINJECTED')
    code, report = invoke(capsys)
    assert code == 1
    assert 'missing\\nINJECTED' in report
    assert '\nINJECTED' not in report


def test_doctor_runs_database_check_only_for_valid_configuration(configuration, monkeypatch, capsys):
    code, report = invoke(capsys)
    assert code == 0
    diagnostics.check_database.assert_called_once()
    diagnostics.check_database.reset_mock()
    monkeypatch.delenv('DOCTOR_TEST_PASSWORD')
    code, report = invoke(capsys)
    assert code == 1
    diagnostics.check_database.assert_not_called()


def test_database_failure_affects_exit_but_does_not_stop_file_checks(configuration, capsys):
    diagnostics.check_database.return_value = [diagnostics.ConfigurationCheck(
        'Database connection', 'INVALID', 'Cannot connect; check host and credentials.')]
    code, report = invoke(capsys)
    assert code == 1
    assert 'Database connection' in report and 'Storage' in report and 'Model catalog' in report
