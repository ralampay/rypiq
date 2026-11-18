import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from app.models.application import Application


@pytest.fixture
def invoke(backend, tmp_path):
    config = tmp_path / 'config/database.yml'
    config.parent.mkdir()
    config.write_text(yaml.safe_dump({'test': backend.settings}))
    storage_config = tmp_path / 'config/storage.yml'
    storage_config.write_text(yaml.safe_dump({'test': {'adapter': 'local', 'root': str(tmp_path / 'objects')}}))
    environment = dict(os.environ, RYPIQ_DATABASE_CONFIG=str(config), DATABASE_ENV='test',
                       STORAGE_ENV='test', RYPIQ_STORAGE_CONFIG=str(storage_config))
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
    result = invoke('applications:create', '--name', 'example', '--embedding-model', 'model',
                    '--provision-type', 'openai', '--embedding-size', '4', '--data',
                    '{"nested":[1],"provider_config":{"embedding":{"api_key_env":"RYPIQ_TEST_MISSING_API_KEY"}}}')
    assert result.returncode == 0, result.stderr
    created = json.loads(result.stdout)
    assert created['embedding_size'] == 4
    assert created['data']['nested'] == [1]
    assert created['api_key']
    for result in (invoke('applications:list', '--output', 'json'), invoke('applications:show', created['id'])):
        assert result.returncode == 0, result.stderr
        payload = json.loads(result.stdout)
        records = payload if isinstance(payload, list) else [payload]
        assert all('api_key' not in record for record in records)
        assert created['api_key'] not in result.stderr
    assert invoke('search', 'query').returncode != 0
    unsupported = invoke('search', 'query', '--application-id', created['id'])
    assert unsupported.returncode != 0
    assert 'credential' in unsupported.stderr
    invalid = invoke('applications:create', '--name', 'bad', '--embedding-model', 'model', '--data', '[]')
    assert invalid.returncode != 0
    assert len(json.loads(invoke('applications:list', '--output', 'json').stdout)) == 1


def test_applications_list_installed_cli_lists_all_records(invoke, backend):
    backend.create_database()
    empty = invoke('applications:list', '--output', 'json')
    assert empty.returncode == 0, empty.stderr
    assert json.loads(empty.stdout) == []
    empty_text = invoke('applications:list')
    assert empty_text.returncode == 0, empty_text.stderr
    assert empty_text.stdout == f'{"ID":<36}  NAME\n'

    repository = backend.applications()
    applications = [repository.create(Application(name=f'Application {index}',
                    embedding_model='unused')) for index in range(101)]
    applications.sort(key=lambda value: value.id)
    result = invoke('applications:list', '--output', 'json')
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        {'id': str(value.id), 'name': value.name} for value in applications]
    for arguments in ((), ('--output', 'text')):
        result = invoke('applications:list', *arguments)
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines() == [f'{"ID":<36}  NAME'] + [
            f'{value.id}  {value.name}' for value in applications]


def test_application_rename_and_delete_preserve_related_data(invoke, backend):
    from app.models.document import Document

    backend.create_database()
    repository = backend.applications()
    value = repository.create(Application(name='Original', embedding_model='model', embedding_size=3))
    document = backend.documents().create(Document(
        content='Keep this document', embedding=[1., 0., 0.], application_id=value.id,
        embedding_model=value.embedding_model, embedding_size=3,
        provision_type=value.provision_type))

    renamed = invoke('applications:update', str(value.id), '--name', 'Renamed')
    assert renamed.returncode == 0, renamed.stderr
    assert json.loads(renamed.stdout)['name'] == 'Renamed'
    assert 'api_key' not in json.loads(renamed.stdout)
    loaded = repository.get(value.id)
    assert loaded.name == 'Renamed'
    assert loaded.api_key == value.api_key
    assert loaded.embedding_model == value.embedding_model

    rejected = invoke('applications:delete', str(value.id))
    assert rejected.returncode != 0
    assert 'Cannot delete application while it has documents' in rejected.stderr
    assert rejected.stdout == ''
    assert repository.get(value.id) is not None
    assert backend.documents().get(document.id) is not None

    empty = repository.create(Application(name='Unused', embedding_model='model', embedding_size=3))
    for expected in (True, False):
        deleted = invoke('applications:delete', str(empty.id))
        assert deleted.returncode == 0, deleted.stderr
        assert json.loads(deleted.stdout) == {'deleted': expected}
        assert repository.get(empty.id) is None


def test_embedding_cli_lifecycle_with_remote_provider(invoke, backend, server, tmp_path):
    from uuid import UUID

    backend.create_database()
    value = backend.applications().create(Application(
        name='remote', embedding_model='remote-model', provision_type='llama-server', embedding_size=2,
        data={'chunking': {'chunk_size': 5, 'stride': 5},
              'provider_config': {'embedding': {'base_url': server['url'] + '/v1'}}}))
    source = tmp_path / 'source.txt'
    source.write_text('abcdefghij')
    output = tmp_path / 'created.json'
    created = invoke('embeddings:create', str(source), '--application-id', str(value.id), '--output', str(output))
    assert created.returncode == 0, created.stderr
    result = json.loads(created.stdout)
    assert result == json.loads(output.read_text())
    assert result['indexed_chunks'] == 2
    first, second = result['chunks']
    assert first['embedding'] == [1., 0.]
    assert backend.documents().get(UUID(first['id'])).content == 'abcde'
    assert backend.documents().get(UUID(second['id'])).content == 'fghij'
    stored = tmp_path / 'objects' / result['storage_key']
    assert stored.read_text() == source.read_text()

    # Changing future chunking must not alter already-persisted chunks.
    updated = invoke('applications:update', str(value.id), '--chunk-size', '10', '--stride', '10')
    assert updated.returncode == 0, updated.stderr
    single = tmp_path / 'shown.json'
    shown = invoke('embeddings:show', '--id', first['id'], '--output', str(single))
    assert shown.returncode == 0, shown.stderr
    assert json.loads(shown.stdout) == json.loads(single.read_text()) == first
    requests_before = len(server['requests'])
    for expected in (True, False):
        deleted = invoke('embeddings:delete', '--id', first['id'])
        assert deleted.returncode == 0, deleted.stderr
        assert json.loads(deleted.stdout) == {'deleted': expected}
    assert invoke('embeddings:show', '--id', first['id']).returncode != 0
    assert backend.documents().get(UUID(second['id'])) is not None
    assert stored.read_text() == source.read_text()
    assert len(server['requests']) == requests_before
