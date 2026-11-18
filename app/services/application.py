"""Application CRUD; callers own the injected repository and its resources."""
from uuid import UUID

from app.exceptions import ApplicationNotFoundError
from app.database.repositories.application import ApplicationRepository
from app.models.application import Application, validate_application
from app.models.validation import validate_id, validate_pagination


class ApplicationService:
    def __init__(self, *, repository: ApplicationRepository) -> None:
        self.repository = repository

    def create(self, application: Application) -> Application:
        validate_id(application.id)
        validate_application(application)
        return self.repository.create(application)

    def get(self, application_id: UUID) -> Application:
        validate_id(application_id)
        application = self.repository.get(application_id)
        if application is None:
            raise ApplicationNotFoundError(application_id, resource_id=application_id)
        return application

    def list(self, *, limit: int = 100, offset: int = 0) -> list[Application]:
        validate_pagination(limit, offset)
        return self.repository.list(limit=limit, offset=offset)

    def update(self, application: Application) -> Application:
        validate_id(application.id)
        validate_application(application)
        return self.repository.update(application)

    def delete(self, application_id: UUID) -> bool:
        validate_id(application_id)
        return self.repository.delete(application_id)
