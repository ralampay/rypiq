import json
from unittest.mock import MagicMock, Mock
from uuid import uuid4

import pytest

from app import application, cli
from app.models.application import Application, validate_application
from app.models.document import Document
import app.storage.factory


@pytest.fixture
def resources(monkeypatch, tmp_path):
    value = Application(name='test', embedding_model='fake', embedding_size=8,
                        data={'chunking': {'chunk_size': 5, 'stride': 5}})
    backend = MagicMock()
    backend.__enter__.return_value = backend
    backend.applications.return_value.get.return_value = value
    backend.settings = {'dimensions': 8}
    repository = backend.documents.return_value
    repository.create.side_effect = lambda record: record
    repository.delete.return_value = True
    monkeypatch.setattr(application, 'get_database', lambda: backend)
    provider = Mock(model_name='fake')
    provider.embed.side_effect = lambda texts: [[float(i) for i in range(8)] for text in texts]
    monkeypatch.setattr(application, 'get_embedding_provider', lambda **kwargs: provider)
    storage = Mock()
    storage.save.return_value = Mock(key='stored.txt', content_type='text/plain', checksum='checksum')
    monkeypatch.setattr(app.storage.factory, 'get_storage', lambda: storage)
    source = tmp_path / 'input.txt'
    source.write_text('abcdefghij')
    return value, backend, repository, provider, storage, source


def create_args(resources):
    value, _, _, _, _, source = resources
    return ['embeddings:create', str(source), '--application-id', str(value.id)]


@pytest.mark.parametrize('full', [False, True])
def test_create_persists_and_exports_full_vectors(resources, tmp_path, capsys, full):
    value, backend, repository, provider, storage, _ = resources
    output = tmp_path / 'vectors.json'
    output.write_text('old export')
    cli.main(create_args(resources) + ['--output', str(output)] + (['--embeddings-full'] if full else []))
    console = json.loads(capsys.readouterr().out)
    exported = json.loads(output.read_text())
    assert exported['indexed_chunks'] == 2
    assert exported['storage_key'] == 'stored.txt'
    assert len(exported['chunks']) == repository.create.call_count == 2
    for record, saved in zip(exported['chunks'], repository.create.call_args_list):
        assert record['id'] == str(saved.args[0].id)
        assert record['application_id'] == str(value.id)
        assert record['embedding'] == list(range(8))
        assert record['metadata']['chunk_size'] == 5
    assert console['chunks'][0]['embedding'] == (list(range(8)) if full else [0, 1, 2, '...', 5, 6, 7])
    backend.applications.return_value.get.assert_called_once_with(value.id)
    storage.save.assert_called_once()
    provider.close.assert_called_once()
    backend.__exit__.assert_called_once()


def test_create_defaults_and_no_export(resources, capsys):
    value, _, repository, _, _, source = resources
    value.data = {}
    cli.main(create_args(resources))
    result = json.loads(capsys.readouterr().out)
    assert result['indexed_chunks'] == 1
    assert repository.create.call_args.args[0].metadata['chunk_size'] == 500
    assert repository.create.call_args.args[0].metadata['stride'] == 400
    assert list(source.parent.iterdir()) == [source]


@pytest.mark.parametrize('options', [[], ['--embeddings-full']])
def test_show_exports_without_loading_provider_or_storage(resources, capsys, tmp_path, options):
    value, backend, repository, provider, storage, _ = resources
    record = Document(content='text', embedding=list(range(8)), application_id=value.id)
    repository.get.return_value = record
    output = tmp_path / 'one.json'
    cli.main(['embeddings:show', '--id', str(record.id), '--output', str(output), *options])
    assert json.loads(output.read_text())['embedding'] == list(range(8))
    assert json.loads(capsys.readouterr().out)['embedding'] == (list(range(8)) if options else [0, 1, 2, '...', 5, 6, 7])
    repository.get.assert_called_once_with(record.id)
    provider.embed.assert_not_called()
    storage.save.assert_not_called()
    backend.__exit__.assert_called_once()


def test_show_missing_is_error(resources, capsys):
    _, backend, repository, _, _, _ = resources
    repository.get.return_value = None
    with pytest.raises(SystemExit) as exc:
        cli.main(['embeddings:show', '--id', str(uuid4())])
    assert exc.value.code == 2
    assert 'not found' in capsys.readouterr().err
    backend.__exit__.assert_called_once()


@pytest.mark.parametrize('found', [True, False])
def test_delete_one_chunk_preserves_storage(resources, capsys, found):
    _, backend, repository, provider, storage, _ = resources
    repository.delete.return_value = found
    identifier = uuid4()
    cli.main(['embeddings:delete', '--id', str(identifier)])
    assert json.loads(capsys.readouterr().out) == {'deleted': found}
    repository.delete.assert_called_once_with(identifier)
    storage.delete.assert_not_called()
    provider.embed.assert_not_called()
    backend.__exit__.assert_called_once()


@pytest.mark.parametrize('options', [[], ['--application-id', 'invalid']])
def test_create_requires_valid_application_id_before_io(resources, options):
    _, backend, _, provider, storage, source = resources
    with pytest.raises(SystemExit):
        cli.main(['embeddings:create', str(source), *options])
    backend.__enter__.assert_not_called()
    provider.embed.assert_not_called()
    storage.save.assert_not_called()


@pytest.mark.parametrize('command', ['embed', 'ingest'])
def test_removed_commands_are_rejected(command, capsys):
    with pytest.raises(SystemExit):
        cli.main([command])
    assert 'invalid choice' in capsys.readouterr().err


@pytest.mark.parametrize('chunking', [None, [], {'chunk_size': True}, {'chunk_size': 0},
                                     {'stride': -1}, {'chunk_size': 10, 'stride': 20}])
def test_application_rejects_invalid_chunking(chunking):
    value = Application(name='test', embedding_model='fake', data={'chunking': chunking})
    with pytest.raises(ValueError, match='[Cc]hunk|[Ss]tride'):
        validate_application(value)


def test_invalid_saved_settings_fail_before_provider_and_storage(resources):
    value, backend, repository, provider, storage, _ = resources
    value.data = {'chunking': {'chunk_size': 0}}
    with pytest.raises(SystemExit):
        cli.main(create_args(resources))
    provider.embed.assert_not_called()
    storage.save.assert_not_called()
    repository.create.assert_not_called()
    backend.__exit__.assert_called_once()


@pytest.mark.parametrize('destination', ['same', 'missing_parent', 'directory'])
def test_bad_export_path_fails_before_persistence(resources, tmp_path, destination):
    _, backend, repository, provider, _, source = resources
    output = {'same': source, 'missing_parent': tmp_path / 'missing' / 'out.json', 'directory': tmp_path}[destination]
    with pytest.raises(SystemExit):
        cli.main(create_args(resources) + ['--output', str(output)])
    repository.create.assert_not_called()
    provider.embed.assert_not_called()
    backend.__enter__.assert_not_called()
    assert source.read_text() == 'abcdefghij'


def test_export_replace_failure_retains_persisted_records(resources, tmp_path, monkeypatch, capsys):
    import os
    _, _, repository, _, storage, source = resources
    output = tmp_path / 'out.json'
    output.write_text('previous')
    monkeypatch.setattr(os, 'replace', Mock(side_effect=OSError('private OS detail')))
    with pytest.raises(SystemExit) as exc:
        cli.main(create_args(resources) + ['--output', str(output)])
    assert exc.value.code == 2
    message = capsys.readouterr().err
    assert 'persisted' in message and 'export failed' in message
    assert all(str(c.args[0].id) in message for c in repository.create.call_args_list)
    assert repository.create.call_count == 2
    repository.delete.assert_not_called()
    storage.delete.assert_not_called()
    assert output.read_text() == 'previous'
    assert set(tmp_path.iterdir()) == {source, output}


def test_ingestion_failure_compensates_and_does_not_export(resources, tmp_path):
    _, backend, repository, provider, storage, _ = resources
    repository.create.side_effect = [None, ValueError('write failed')]
    output = tmp_path / 'out.json'
    with pytest.raises(SystemExit):
        cli.main(create_args(resources) + ['--output', str(output)])
    assert not output.exists()
    repository.delete.assert_called_once_with(repository.create.call_args_list[0].args[0].id)
    storage.delete.assert_called_once_with('stored.txt')
    provider.close.assert_called_once()
    backend.__exit__.assert_called_once()


@pytest.mark.parametrize('data_option', [False, True])
def test_application_chunk_flags_merge_without_mutation(resources, capsys, data_option):
    value, backend, _, _, _, _ = resources
    original = {'keep': 1, 'chunking': {'chunk_size': 5, 'stride': 5}}
    value.data = original
    backend.applications.return_value.update.side_effect = lambda record: record
    args = ['applications:update', str(value.id), '--chunk-size', '20']
    if data_option:
        args += ['--data', '{"new":2,"chunking":{"chunk_size":30,"stride":10}}']
    cli.main(args)
    result = json.loads(capsys.readouterr().out)
    assert result['data'] == ({'new': 2, 'chunking': {'chunk_size': 20, 'stride': 10}}
                              if data_option else {'keep': 1, 'chunking': {'chunk_size': 20, 'stride': 5}})
    assert value.data == original == {'keep': 1, 'chunking': {'chunk_size': 5, 'stride': 5}}


def test_application_creation_chunk_flags_override_data(resources, capsys):
    _, backend, _, _, _, _ = resources
    backend.applications.return_value.create.side_effect = lambda value: value
    cli.main(['applications:create', '--name', 'test', '--embedding-model', 'fake',
              '--data', '{"keep":1,"chunking":{"chunk_size":30,"stride":10}}',
              '--chunk-size', '20', '--stride', '15'])
    assert json.loads(capsys.readouterr().out)['data'] == {'keep': 1, 'chunking': {'chunk_size': 20, 'stride': 15}}


@pytest.mark.parametrize('options', [['--model', 'other'], ['--chunk_size', '10'],
                                    ['--stride', '5'], ['--context_size', '10']])
def test_creation_disallows_application_overrides(resources, options):
    _, backend, _, _, _, _ = resources
    with pytest.raises(SystemExit):
        cli.main(create_args(resources) + options)
    backend.__enter__.assert_not_called()


@pytest.mark.parametrize('command', ['embeddings:show', 'embeddings:delete'])
def test_record_commands_require_valid_id(resources, command):
    _, backend, _, _, _, _ = resources
    for options in ([], ['--id', 'bad']):
        with pytest.raises(SystemExit):
            cli.main([command, *options])
    backend.__enter__.assert_not_called()


def test_short_vector_shown_completely(resources, capsys):
    _, _, repository, _, _, _ = resources
    repository.get.return_value = Document(content='text', embedding=[1., 0.])
    cli.main(['embeddings:show', '--id', str(uuid4())])
    assert json.loads(capsys.readouterr().out)['embedding'] == [1., 0.]


@pytest.mark.parametrize('command', ['embeddings:show', 'embeddings:delete'])
def test_record_failure_is_sanitized_and_releases_database(resources, capsys, command):
    from sqlalchemy.exc import SQLAlchemyError
    _, backend, repository, provider, storage, _ = resources
    repository.get.side_effect = repository.delete.side_effect = SQLAlchemyError('private driver detail')
    with pytest.raises(SystemExit):
        cli.main([command, '--id', str(uuid4())])
    result = capsys.readouterr()
    assert result.out == ''
    assert 'Database operation failed' in result.err
    assert 'private driver detail' not in result.err
    backend.__exit__.assert_called_once()
    provider.embed.assert_not_called()
    storage.delete.assert_not_called()


def test_output_hard_link_cannot_replace_input(resources, tmp_path):
    _, backend, _, _, _, source = resources
    output = tmp_path / 'linked.json'
    output.hardlink_to(source)
    with pytest.raises(SystemExit):
        cli.main(create_args(resources) + ['--output', str(output)])
    backend.__enter__.assert_not_called()


def test_missing_chunk_defaults_are_resolved_independently():
    from app.models.application import chunking_settings
    assert chunking_settings({'chunking': {'chunk_size': 600}}) == (600, 400)
    assert chunking_settings({'chunking': {'stride': 250}}) == (500, 250)


def test_export_write_failure_leaves_previous_file_and_cleans_temporary(tmp_path, monkeypatch):
    from app.cli_output import write_json_export
    import tempfile
    destination = tmp_path / 'existing.json'
    destination.write_text('previous')
    real_factory = tempfile.NamedTemporaryFile

    def broken_file(**kwargs):
        stream = real_factory(**kwargs)
        stream.write = Mock(side_effect=OSError('disk full'))
        return stream

    monkeypatch.setattr(tempfile, 'NamedTemporaryFile', broken_file)
    with pytest.raises(OSError):
        write_json_export(destination, {'ok': True})
    assert destination.read_text() == 'previous'
    assert list(tmp_path.iterdir()) == [destination]


def test_missing_application_does_not_embed_or_export(resources, tmp_path):
    _, backend, repository, provider, storage, _ = resources
    backend.applications.return_value.get.return_value = None
    output = tmp_path / 'absent.json'
    with pytest.raises(SystemExit):
        cli.main(create_args(resources) + ['--output', str(output)])
    provider.embed.assert_not_called()
    storage.save.assert_not_called()
    repository.create.assert_not_called()
    backend.__exit__.assert_called_once()
    assert not output.exists()


def test_invalid_batch_size_rejected_before_embedding(resources):
    _, backend, repository, provider, storage, _ = resources
    with pytest.raises(SystemExit):
        cli.main(create_args(resources) + ['--batch_size', '0'])
    provider.embed.assert_not_called()
    storage.save.assert_not_called()
    repository.create.assert_not_called()
    backend.__exit__.assert_called_once()
