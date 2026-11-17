import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml


@pytest.fixture
def invoke(backend, tmp_path):
    config = tmp_path / 'config/database.yml'
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({'test': backend.settings}))
    environment = dict(os.environ, RYPIQ_DATABASE_CONFIG=str(config), DATABASE_ENV='test')
    executable = Path(sys.executable).parent / 'rypiq'
    def run(command, *arguments):
        return subprocess.run([str(executable), command, *arguments], cwd=tmp_path, env=environment,
                              capture_output=True, text=True, timeout=30)
    return run


def test_installed_cli_lifecycle_outside_repository(invoke):
    missing = invoke('db:migrate')
    assert missing.returncode != 0
    assert 'db:create' in missing.stderr
    for command, expected in [
        ('db:create', {'created': True, 'tables_created': ['applications', 'documents']}),
        ('db:create', {'created': False, 'tables_created': []}),
        ('db:migrate', {'tables_created': []}),
        ('db:reset', {'dropped': True, 'created': True, 'tables_created': ['applications', 'documents']}),
        ('db:drop', {'dropped': True}),
        ('db:drop', {'dropped': False}),
    ]:
        result = invoke(command)
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == expected


@pytest.mark.parametrize('command', ['db:upgrade', 'db:rollback', 'db:status', 'db:migration'])
def test_removed_cli_commands_are_rejected(invoke, command):
    result = invoke(command)
    assert result.returncode != 0
    assert 'invalid choice' in result.stderr


def test_help_lists_only_supported_database_commands(invoke):
    result = invoke('--help')
    assert result.returncode == 0
    for command in ('db:create', 'db:migrate', 'db:drop', 'db:reset'):
        assert command in result.stdout
    assert 'db:upgrade' not in result.stdout


def test_application_management_and_required_scope(invoke):
    assert invoke('db:create').returncode == 0
    result = invoke('application:create', '--name', 'example', '--embedding-model', 'model',
                    '--provision-type', 'openai', '--embedding-size', '4', '--data',
                    '{"nested":[1],"provider_config":{"embedding":{"api_key_env":"RYPIQ_TEST_MISSING_API_KEY"}}}')
    assert result.returncode == 0, result.stderr
    created = json.loads(result.stdout)
    assert created['embedding_size'] == 4
    assert created['data']['nested'] == [1]
    assert created['api_key']
    for result in (invoke('application:list'), invoke('application:show', created['id'])):
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        records = payload if isinstance(payload, list) else [payload]
        assert all('api_key' not in record for record in records)
        assert created['api_key'] not in result.stderr
    assert invoke('search', 'query').returncode != 0
    unsupported = invoke('search', 'query', '--application-id', created['id'])
    assert unsupported.returncode != 0
    assert 'credential' in unsupported.stderr
    invalid = invoke('application:create', '--name', 'bad', '--embedding-model', 'model', '--data', '[]')
    assert invalid.returncode != 0
    assert len(json.loads(invoke('application:list').stdout)) == 1
