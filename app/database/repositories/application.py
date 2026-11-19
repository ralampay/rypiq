from __future__ import annotations

from abc import ABC, abstractmethod
from uuid import UUID

from app.models.application import Application, ApplicationSummary


class ApplicationRepository(ABC):
    @abstractmethod
    def create(self, application: Application) -> Application:
        raise NotImplementedError

    @abstractmethod
    def get(self, id: UUID) -> Application | None:
        raise NotImplementedError

    @abstractmethod
    def list(self, limit: int = 100, offset: int = 0) -> list[Application]:
        raise NotImplementedError

    @abstractmethod
    def update(self, application: Application) -> Application:
        raise NotImplementedError

    @abstractmethod
    def delete(self, id: UUID) -> bool:
        raise NotImplementedError

    @abstractmethod
    def list_summaries(self, *, limit: int = 100, offset: int = 0) -> list[ApplicationSummary]:
        """List application configuration and counts of associated documents."""
        raise NotImplementedError
