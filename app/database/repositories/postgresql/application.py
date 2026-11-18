from uuid import UUID
from dataclasses import asdict
from datetime import datetime, timezone

import sqlalchemy as sa

from app.database.repositories.application import ApplicationRepository
from app.database.repositories.postgresql._validation import positive_limit
from app.models.application import Application, validate_application
from app.providers.config import embedding_preprocessing


class PostgreSQLApplicationRepository(ApplicationRepository):
    def __init__(self, engine, table):
        self.engine = engine
        self.table = table

    def create(self, application: Application) -> Application:
        validate_application(application)
        with self.engine.begin() as connection:
            connection.execute(self.table.insert().values(**asdict(application)))
        return application

    def get(self, id: UUID) -> Application | None:
        with self.engine.connect() as connection:
            row = connection.execute(sa.select(self.table).where(self.table.c.id == id)).mappings().first()
        return Application(**row) if row else None

    def list(self, limit: int = 100, offset: int = 0) -> list[Application]:
        positive_limit(limit)
        if type(offset) is not int or offset < 0:
            raise ValueError('Offset must be a nonnegative integer')
        with self.engine.connect() as connection:
            rows = connection.execute(sa.select(self.table).order_by(
                self.table.c.id).limit(limit).offset(offset)).mappings().all()
        return [Application(**row) for row in rows]

    def update(self, application: Application) -> Application:
        validate_application(application)
        with self.engine.begin() as connection:
            original = connection.execute(sa.select(self.table).where(
                self.table.c.id == application.id).with_for_update()).mappings().first()
            if original is None:
                raise KeyError(application.id)
            if application.api_key != original['api_key'] or application.created_at != original['created_at']:
                raise ValueError('Application API key and creation time are immutable')
            changed_embedding = any(getattr(application, name) != original[name] for name in (
                'embedding_model', 'embedding_size', 'provision_type'))
            changed_embedding |= embedding_preprocessing(application.data, application.provision_type, application.embedding_size) != embedding_preprocessing(original['data'], original['provision_type'], original['embedding_size'])
            if changed_embedding and self._has_documents(connection, application.id):
                raise ValueError('Cannot change embedding settings while application has documents')
            values = asdict(application)
            values['updated_at'] = datetime.now(timezone.utc)
            connection.execute(self.table.update().where(self.table.c.id == application.id).values(**values))
        application.updated_at = values['updated_at']
        return application

    def _has_documents(self, connection, id: UUID) -> bool:
        documents = self.table.metadata.tables['public.documents']
        return bool(connection.execute(sa.select(sa.exists().where(documents.c.application_id == id))).scalar())

    def delete(self, id: UUID) -> bool:
        with self.engine.begin() as connection:
            original = connection.execute(sa.select(self.table.c.id).where(
                self.table.c.id == id).with_for_update()).first()
            if original is None:
                return False
            if self._has_documents(connection, id):
                raise ValueError('Cannot delete application while it has documents')
            connection.execute(self.table.delete().where(self.table.c.id == id))
        return True
