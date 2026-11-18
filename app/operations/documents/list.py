"""Application-scoped document summaries; no chunk vectors or file access."""
from uuid import UUID
from app.database.repositories.base import ApplicationRepository, DocumentRepository
from app.models.document import DocumentSummary
from app.operations.applications.get import get_application


def list_documents(*, application_id: UUID, repository: DocumentRepository,
                   applications: ApplicationRepository) -> list[DocumentSummary]:
    get_application(application_id=application_id, repository=applications)
    summaries: list[DocumentSummary] = []
    while True:
        page = repository.list_summaries(application_id=application_id, limit=100, offset=len(summaries))
        summaries.extend(page)
        if len(page) < 100:
            return summaries
