from uuid import UUID

from app.database.repositories.application import ApplicationRepository
from app.models.application import Application
from app.services.application import ApplicationService


def get_application(*, application_id: UUID, repository: ApplicationRepository) -> Application:
    return ApplicationService(repository=repository).get(application_id)
