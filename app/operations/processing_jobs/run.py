"""Execute one claimed attempt; the caller owns providers and heartbeat resources."""
from collections.abc import Callable
from contextlib import AbstractContextManager

from app.database.repositories.processing_job import ProcessingJobRepository
from app.exceptions import ProcessingBusyError
from app.logging import get_logger
from app.models.processing_job import ProcessingJob


class RunProcessingJob:
    def __init__(
        self,
        *,
        repository: ProcessingJobRepository,
        process: Callable[[ProcessingJob], object],
        heartbeat: Callable[[ProcessingJob], AbstractContextManager],
    ) -> None:
        self.repository = repository
        self.process = process
        self.heartbeat = heartbeat

    def execute(self) -> bool:
        job = self.repository.claim()
        if job is None:
            return False
        with self.heartbeat(job):
            try:
                self.process(job)
            except ProcessingBusyError:
                self.repository.release(job)
            except Exception as failure:
                # Never persist arbitrary exception messages or source contents.
                self.repository.fail(
                    job, message='Document processing failed; check configuration and retry.',
                )
                get_logger('worker').error(
                    'processing_failed',
                    job_id=str(job.id),
                    error_type=type(failure).__name__,
                )
        return True
