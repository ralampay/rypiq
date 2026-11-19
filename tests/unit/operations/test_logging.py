import json
import logging

import pytest


def test_logging_configuration_and_context(tmp_path, monkeypatch):
    from app.logging import configure_logging, get_logger, log_context, shutdown_logging

    monkeypatch.setenv('LOG_ENV', 'test')
    configure_logging(directory=tmp_path, format='json')
    try:
        with log_context(request_id='request-1'):
            get_logger('documents').info('stored', document_id='doc-1')
        logging.getLogger('rypiq.other').info('ordinary')
        configure_logging(directory=tmp_path, format='json')
        get_logger().info('once')
    finally:
        shutdown_logging()
    records = [json.loads(line) for line in (tmp_path / 'test.log').read_text().splitlines()]
    assert [record['event'] for record in records] == ['stored', 'ordinary', 'once']
    assert records[0]['request_id'] == 'request-1'
    assert 'request_id' not in records[1]
    assert records[0]['environment'] == 'test'


def test_unconfigured_library_is_silent(capsys):
    from app.logging import get_logger, shutdown_logging

    shutdown_logging()
    get_logger().error('silent')
    assert capsys.readouterr() == ('', '')


def test_bad_environment_has_no_filesystem_effect(tmp_path):
    from app.logging import configure_logging
    from app.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        configure_logging(environment='../escape', directory=tmp_path / 'logs')
    assert not (tmp_path / 'logs').exists()


def test_cli_logs_outcome_without_changing_json(tmp_path, monkeypatch, capsys):
    from app import cli

    monkeypatch.setenv('RYPIQ_LOG_DIR', str(tmp_path))
    monkeypatch.setenv('LOG_ENV', 'test')
    monkeypatch.setenv('RYPIQ_LOG_FORMAT', 'json')
    cli.main(['process', 'text', 'Hello world'])
    assert json.loads(capsys.readouterr().out)['words'] == 2
    records = [json.loads(line) for line in (tmp_path / 'test.log').read_text().splitlines()]
    assert [record['event'] for record in records] == ['operation_started', 'operation_completed']
    assert records[0]['operation'] == 'cli'


def test_search_failure_logs_type_without_query_or_diagnostics(tmp_path):
    from app.logging import configure_logging, shutdown_logging
    from app.operations.documents.search import SearchDocuments

    configure_logging(environment='test', directory=tmp_path, format='json')
    try:
        with pytest.raises(ValueError):
            SearchDocuments(query='', embedding_provider=None, repository=None, application=None).execute()
    finally:
        shutdown_logging()
    records = [json.loads(line) for line in (tmp_path / 'test.log').read_text().splitlines()]
    assert records[-1]['event'] == 'operation_failed'
    assert records[-1]['error_type'] == 'ValidationError'
    assert 'Query cannot be empty' not in (tmp_path / 'test.log').read_text()


@pytest.fixture(autouse=True)
def reset_logging(monkeypatch):
    from app.logging import shutdown_logging

    for name in ('LOG_ENV', 'DATABASE_ENV', 'RYPIQ_LOGGING_CONFIG', 'RYPIQ_LOG_DIR',
                 'RYPIQ_LOG_LEVEL', 'RYPIQ_LOG_FORMAT', 'RYPIQ_LOG_CONSOLE'):
        monkeypatch.delenv(name, raising=False)
    shutdown_logging()
    yield
    shutdown_logging()


def test_settings_precedence_and_defaults(tmp_path, monkeypatch):
    from app.logging import load_logging_settings

    assert load_logging_settings().directory == tmp_path / 'logs'
    assert load_logging_settings().environment == 'development'
    config = tmp_path / 'logging.yml'
    config.write_text('production:\n  level: WARNING\n  format: json\n  directory: custom\n')
    monkeypatch.setenv('RYPIQ_LOGGING_CONFIG', str(config))
    monkeypatch.setenv('DATABASE_ENV', 'production')
    assert load_logging_settings().level == 'WARNING'
    assert load_logging_settings().directory == tmp_path / 'custom'
    monkeypatch.setenv('RYPIQ_LOG_LEVEL', 'error')
    assert load_logging_settings().level == 'ERROR'
    assert load_logging_settings(level='debug').level == 'DEBUG'
    monkeypatch.setenv('LOG_ENV', 'test')
    assert load_logging_settings().environment == 'test'
    assert load_logging_settings(environment='development').environment == 'development'


@pytest.mark.parametrize('contents', ['[1]', 'test: [1]', 'test:\n  unknown: x', 'test:\n  level: 42', 'test:\n  console: yesplease', 'test:\n  format: xml'])
def test_invalid_yaml_settings(tmp_path, contents):
    from app.logging import load_logging_settings
    from app.exceptions import ConfigurationError

    config = tmp_path / 'logging.yml'
    config.write_text(contents)
    with pytest.raises(ConfigurationError):
        load_logging_settings(environment='test', config_path=config)


def test_missing_explicit_config(tmp_path):
    from app.logging import load_logging_settings
    from app.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError):
        load_logging_settings(config_path=tmp_path / 'absent')


def test_text_filter_console_and_host_preservation(tmp_path, capsys):
    import structlog
    from app.logging import configure_logging, get_logger, shutdown_logging

    before = structlog.get_config().copy()
    root_handlers = logging.getLogger().handlers[:]
    configure_logging(directory=tmp_path, console=True, level='WARNING')
    get_logger().info('filtered')
    get_logger().warning('visible', count=2)
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'visible' in captured.err and 'filtered' not in captured.err
    text = (tmp_path / 'development.log').read_text()
    assert 'timestamp=' in text and 'count=2' in text and '\x1b' not in text
    shutdown_logging()
    assert logging.getLogger().handlers == root_handlers
    assert structlog.get_config() == before


def test_reconfiguration_and_rotation(tmp_path):
    from app.logging import configure_logging, get_logger

    logger = get_logger()
    configure_logging(directory=tmp_path)
    logger.info('first')
    original = tmp_path / 'development.log'
    archived = tmp_path / 'development.log.1'
    original.rename(archived)
    logger.info('second')
    assert 'second' in original.read_text()
    assert 'second' not in archived.read_text()
    configure_logging(directory=tmp_path, environment='test')
    logger.info('third')
    assert 'third' in (tmp_path / 'test.log').read_text()
    assert 'third' not in original.read_text()


def test_failed_reconfiguration_keeps_existing_output(tmp_path):
    from app.logging import configure_logging, get_logger
    from app.exceptions import ConfigurationError

    configure_logging(directory=tmp_path)
    bad = tmp_path / 'file'
    bad.write_text('not a directory')
    with pytest.raises(ConfigurationError):
        configure_logging(directory=bad)
    get_logger().info('survived')
    assert 'survived' in (tmp_path / 'development.log').read_text()


def test_write_failure_is_safe(tmp_path, monkeypatch, capsys):
    from app.logging import configure_logging, get_logger
    from logging.handlers import WatchedFileHandler

    configure_logging(directory=tmp_path)
    def fail(self):
        raise OSError('secret connection URL')
    monkeypatch.setattr(WatchedFileHandler, 'reopenIfNeeded', fail)
    get_logger().error('event')
    assert capsys.readouterr().err == 'RyPiq could not write a log entry.\n'


def test_nested_async_context_isolation(tmp_path):
    import asyncio
    from app.logging import configure_logging, get_logger, log_context

    configure_logging(directory=tmp_path, format='json')
    async def task(value):
        with log_context(request_id=value):
            await asyncio.sleep(0)
            with log_context(request_id='nested'):
                get_logger().info('nested')
            get_logger().info('outer')
    async def run():
        await asyncio.gather(task('a'), task('b'))
    asyncio.run(run())
    get_logger().info('outside')
    records = [json.loads(line) for line in (tmp_path / 'development.log').read_text().splitlines()]
    assert {r['request_id'] for r in records if r['event'] == 'outer'} == {'a', 'b'}
    assert all(r['request_id'] == 'nested' for r in records if r['event'] == 'nested')
    assert 'request_id' not in records[-1]


def test_lifecycle_preserves_failure_and_context(tmp_path):
    from app.logging import configure_logging, log_operation, get_logger

    configure_logging(directory=tmp_path, format='json')
    failure = RuntimeError('password=secret')
    with pytest.raises(RuntimeError) as caught:
        with log_operation('example'):
            raise failure
    assert caught.value is failure
    get_logger().info('outside')
    text = (tmp_path / 'development.log').read_text()
    assert 'secret' not in text
    records = [json.loads(line) for line in text.splitlines()]
    assert records[1]['error_type'] == 'RuntimeError'
    assert 'operation' not in records[-1]


def test_standard_logging_preserves_extra_fields(tmp_path):
    from app.logging import configure_logging

    configure_logging(directory=tmp_path, format='json')
    logging.getLogger('rypiq.example').info('stored', extra={'document_id': '123'})
    record = json.loads((tmp_path / 'development.log').read_text())
    assert record['document_id'] == '123'


def test_doctor_does_not_close_library_logging(tmp_path, monkeypatch, capsys):
    from app import cli
    from app.diagnostics import ConfigurationReport
    from app.logging import configure_logging, get_logger

    configure_logging(directory=tmp_path)
    monkeypatch.setattr(cli, 'check_configuration', lambda: ConfigurationReport('development', ()))
    cli.main(['--doctor'])
    get_logger().info('still_active')
    capsys.readouterr()
    assert 'still_active' in (tmp_path / 'development.log').read_text()


def test_owned_handlers_close_without_closing_host_handler(tmp_path):
    from app.logging import configure_logging, shutdown_logging

    namespace = logging.getLogger('rypiq')
    host = logging.FileHandler(tmp_path / 'host.log')
    namespace.addHandler(host)
    try:
        before = set(namespace.handlers)
        configure_logging(directory=tmp_path)
        owned = set(namespace.handlers) - before
        assert len(owned) == 1
        old_handler = owned.pop()
        configure_logging(directory=tmp_path, environment='test')
        assert old_handler.stream is None
        new_handler = (set(namespace.handlers) - before).pop()
        shutdown_logging()
        assert new_handler.stream is None
        assert host in namespace.handlers and not host.stream.closed
    finally:
        namespace.removeHandler(host)
        host.close()


def test_core_workflows_log_success_without_content(tmp_path):
    from unittest.mock import Mock
    from app.logging import configure_logging
    from app.models.application import Application
    from app.operations.documents.search import SearchDocuments
    from app.operations.embeddings.embed_file import embed_chunks
    from app.operations.inference.chat import Chat
    from app.providers.types.chat import ChatMessage

    configure_logging(directory=tmp_path, format='json')
    provider = Mock(model_name='model')
    provider.embed.return_value = [[1., 0.]]
    application = Application(name='example', embedding_model='model', embedding_size=2)
    repository = Mock()
    repository.search_by_vector.return_value = []
    assert SearchDocuments(query='private-query', embedding_provider=provider,
                           repository=repository, application=application).execute() == []
    result = embed_chunks({'chunks': [{'text': 'private-document'}]}, provider, 1)
    assert result['embedded_chunk_count'] == 1
    expected = provider.chat.return_value
    assert Chat(messages=[ChatMessage(role='user', content='private-prompt')], provider=provider).execute() is expected
    provider.close.assert_not_called()
    text = (tmp_path / 'development.log').read_text()
    assert all(secret not in text for secret in ('private-query', 'private-document', 'private-prompt'))
    records = [json.loads(line) for line in text.splitlines()]
    completed = {r['operation']: r for r in records if r['event'] == 'operation_completed'}
    assert set(completed) == {'search', 'embed', 'chat'}
    assert completed['search']['result_count'] == 0
    assert completed['embed']['chunk_count'] == 1


def test_invalid_config_path_is_safe():
    from app.logging import load_logging_settings
    from app.exceptions import ConfigurationError

    with pytest.raises(ConfigurationError, match='Cannot read logging configuration'):
        load_logging_settings(config_path='private\x00path')
