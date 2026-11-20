"""Worker composition and process lifecycle; each attempt owns its resources."""
from contextlib import contextmanager
import signal
from threading import Event, Thread

from app import application
from app.config import AppConfig, load_config
from app.database.factory import get_database
from app.logging import get_logger
from app.operations.documents.process import ProcessDocument
from app.operations.processing_jobs.run import RunProcessingJob
from app.storage.factory import get_storage


@contextmanager
def heartbeat(repository, job):
    stopped = Event()
    def renew():
        while not stopped.wait(10):
            try:
                if not repository.heartbeat(job):
                    return
            except Exception:
                # Writes remain fenced if a renewal cannot reach PostgreSQL.
                get_logger('worker').warning('heartbeat_failed', job_id=str(job.id))
                return
    thread = Thread(target=renew, name='rypiq-job-heartbeat', daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()
        thread.join()


def run_once(*, config: AppConfig | None = None) -> bool:
    config = load_config() if config is None else config
    with get_database(config) as database:
        jobs = database.processing_jobs()
        def process(job):
            return ProcessDocument(document_id=job.document_id, batch_size=job.batch_size,
                repository=database.processing_documents(job), applications=database.applications(),
                storage=get_storage(config), provider_scope=lambda current: application._open_configured_provider(
                    application._application_settings(current, 'embedding', config=config), None,
                    embedding_factory=application.get_embedding_provider, config=config)).execute()
        return RunProcessingJob(repository=jobs, process=process,
                                heartbeat=lambda job: heartbeat(jobs, job)).execute()


def run_worker(*, once: bool = False, config: AppConfig | None = None) -> None:
    config = load_config() if config is None else config
    if once:
        run_once(config=config)
        return
    stopped = Event()
    previous = {}
    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, lambda *_: stopped.set())
        while not stopped.is_set():
            try:
                run_once(config=config)
            except Exception as failure:
                get_logger('worker').error('worker_poll_failed', error_type=type(failure).__name__)
            stopped.wait(1)
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
