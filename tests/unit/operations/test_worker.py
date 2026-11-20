from contextlib import contextmanager
from unittest.mock import Mock

import pytest

from app import cli
from app.exceptions import ProcessingBusyError
from app.operations.processing_jobs.run import RunProcessingJob


def test_worker_failure_releases_heartbeat_and_sanitizes_error():
    events = []
    job = Mock()
    job.id = 'job-id'
    repository = Mock()
    repository.claim.return_value = job
    @contextmanager
    def heartbeat(_):
        events.append('open')
        try:
            yield
        finally:
            events.append('closed')
    process = Mock(side_effect=RuntimeError('secret'))
    assert RunProcessingJob(repository=repository, process=process, heartbeat=heartbeat).execute()
    assert events == ['open', 'closed']
    assert 'secret' not in repository.fail.call_args.kwargs['message']


def test_busy_document_releases_claim_without_terminal_failure():
    from contextlib import nullcontext
    repository = Mock()
    job = repository.claim.return_value
    command = RunProcessingJob(repository=repository, heartbeat=lambda _: nullcontext(),
                               process=Mock(side_effect=ProcessingBusyError('busy')))
    assert command.execute()
    repository.release.assert_called_once_with(job)
    repository.fail.assert_not_called()


def test_server_cli_uses_factory_and_options(monkeypatch):
    import uvicorn
    run = Mock()
    monkeypatch.setattr(uvicorn, 'run', run)
    cli.main(['server', '--host', '127.0.0.1', '--port', '3001', '--reload'])
    run.assert_called_once_with('app:create_app', factory=True, host='127.0.0.1', port=3001, reload=True)


def test_worker_cli_once_and_cleanup(monkeypatch):
    from app import worker
    run, close = Mock(), Mock()
    monkeypatch.setattr(worker, 'run_worker', run)
    monkeypatch.setattr(cli, 'shutdown_logging', close)
    cli.main(['worker', '--once'])
    run.assert_called_once()
    assert run.call_args.kwargs["once"] is True
    from app.config import AppConfig
    assert isinstance(run.call_args.kwargs["config"], AppConfig)
    close.assert_called_once()


@pytest.mark.parametrize('renewal', [True, False, RuntimeError('private database error')])
def test_heartbeat_thread_renews_and_stops_on_scope_exit(monkeypatch, renewal):
    from app import worker
    stopped = Mock()
    stopped.wait.side_effect = [False, True]
    monkeypatch.setattr(worker, 'Event', lambda: stopped)
    repository = Mock()
    if isinstance(renewal, Exception):
        repository.heartbeat.side_effect = renewal
    else:
        repository.heartbeat.return_value = renewal
    job = Mock(id='job')
    with worker.heartbeat(repository, job):
        pass
    repository.heartbeat.assert_called_once_with(job)
    stopped.set.assert_called_once()


def test_worker_shutdown_finishes_attempt_and_restores_signal_handlers(monkeypatch):
    from app import worker
    handlers, events = {}, []
    def set_signal(signum, handler):
        old = handlers.get(signum, 'previous')
        handlers[signum] = handler
        return old
    def run_once(*, config=None):
        events.append('started')
        handlers[worker.signal.SIGTERM]()
        events.append('finished')
    monkeypatch.setattr(worker.signal, 'signal', set_signal)
    monkeypatch.setattr(worker, 'run_once', run_once)
    worker.run_worker()
    assert events == ['started', 'finished']
    assert set(handlers.values()) == {'previous'}
