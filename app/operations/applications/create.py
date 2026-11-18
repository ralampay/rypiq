from app.database.repositories.base import ApplicationRepository
from app.models.application import Application
from app.services.application import ApplicationService


class CreateApplication:
    def __init__(self, *, application: Application, repository: ApplicationRepository) -> None:
        self.application = application
        self.repository = repository

    def execute(self) -> Application:
        return ApplicationService(repository=self.repository).create(self.application)
