from uuid import UUID

from app.database.repositories.base import ApplicationRepository
from app.services.application import ApplicationService


def delete_application(*, application_id: UUID, repository: ApplicationRepository) -> bool:
    return ApplicationService(repository=repository).delete(application_id)
