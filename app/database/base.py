from abc import ABC, abstractmethod

from app.database.repositories.document import DocumentRepository
from app.database.repositories.application import ApplicationRepository
from app.database.repositories.embedding import EmbeddingRepository
from app.database.repositories.processing_job import ProcessingJobRepository
from app.models.processing_job import ProcessingJob


class DatabaseBackend(ABC):

    @abstractmethod
    def create_database(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def documents(self) -> DocumentRepository:
        raise NotImplementedError

    @abstractmethod
    def embeddings(self) -> EmbeddingRepository:
        raise NotImplementedError

    @abstractmethod
    def applications(self) -> ApplicationRepository:
        raise NotImplementedError

    def processing_jobs(self) -> ProcessingJobRepository:
        """Return the durable document queue repository."""
        raise NotImplementedError

    def processing_documents(self, job: ProcessingJob) -> DocumentRepository:
        """Return document persistence fenced to the supplied worker claim."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError
