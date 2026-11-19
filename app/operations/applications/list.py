from app.database.repositories.application import ApplicationRepository
from app.models.application import Application, ApplicationSummary
from app.services.application import ApplicationService


def list_applications(*, repository: ApplicationRepository, limit: int = 100,
                      offset: int = 0) -> list[Application]:
    return ApplicationService(repository=repository).list(limit=limit, offset=offset)


def list_all_applications(*, repository: ApplicationRepository) -> list[Application]:
    """Read all pages in repository order, borrowing the caller's repository."""
    applications: list[Application] = []
    while True:
        page = list_applications(repository=repository, limit=100, offset=len(applications))
        applications.extend(page)
        if len(page) < 100:
            return applications


def list_application_summaries(*, repository: ApplicationRepository) -> list[ApplicationSummary]:
    """Read every application with its document count, borrowing the repository."""
    summaries: list[ApplicationSummary] = []
    while True:
        page = repository.list_summaries(limit=100, offset=len(summaries))
        summaries.extend(page)
        if len(page) < 100:
            return summaries
