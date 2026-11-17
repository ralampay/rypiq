"""SQLAlchemy Core CRUD and exact pgvector cosine search."""
from uuid import UUID

import math
from dataclasses import asdict
from datetime import datetime, timezone

import sqlalchemy as sa

from app.database.repositories.base import DocumentRepository, ApplicationRepository
from app.models.application import Application, validate_application
from app.models.embedding import validate_embedding
from app.models.document import Document
from app.models.search import SearchResult
from app.providers.config import embedding_preprocessing


def positive_limit(limit):
    if type(limit) is not int or limit < 1:
        raise ValueError('Limit must be a positive integer')


class PostgreSQLDocumentRepository(DocumentRepository):
    def __init__(self, engine, table):
        self.engine = engine
        self.table = table
        self.applications = table.metadata.tables["public.applications"]

    def _values(self, document):
        values = asdict(document)
        values['embedding'] = validate_embedding(document.embedding, document.embedding_size)
        if not isinstance(document.application_id, UUID):
            raise ValueError('Document requires an application ID')
        return values

    def _validate_application(self, connection, document):
        row = connection.execute(sa.select(self.applications).where(
            self.applications.c.id == document.application_id).with_for_update(read=True)).mappings().first()
        if row is None:
            raise KeyError(document.application_id)
        for name in ('embedding_model', 'embedding_size', 'provision_type'):
            if getattr(document, name) != row[name]:
                raise ValueError('Document embedding settings must match its application')

    @staticmethod
    def _document(row):
        values = dict(row)
        values['embedding'] = [float(value) for value in values['embedding']]
        return Document(**values)

    def create(self, document):
        values = self._values(document)
        with self.engine.begin() as connection:
            self._validate_application(connection, document)
            connection.execute(self.table.insert().values(**values))
        return document

    def _scope(self, statement, application_id):
        if application_id is not None:
            statement = statement.where(self.table.c.application_id == application_id)
        return statement

    def get(self, id, *, application_id=None):
        with self.engine.connect() as connection:
            row = connection.execute(self._scope(sa.select(self.table).where(
                self.table.c.id == id), application_id)).mappings().first()
        return self._document(row) if row else None

    def update(self, document, *, application_id=None):
        if application_id is not None and document.application_id != application_id:
            raise ValueError('Document does not match application scope')
        values = self._values(document)
        values['updated_at'] = datetime.now(timezone.utc)
        with self.engine.begin() as connection:
            self._validate_application(connection, document)
            original = connection.execute(self._scope(sa.select(self.table).where(
                self.table.c.id == document.id), application_id).with_for_update()).mappings().first()
            if original is None:
                raise KeyError(document.id)
            if any(values[key] != original[key] for key in (
                    'application_id', 'embedding_model', 'embedding_size', 'provision_type')):
                raise ValueError('Document ownership and embedding snapshots cannot be changed')
            result = connection.execute(self.table.update().where(
                self.table.c.id == document.id).values(**values))
            if not result.rowcount:
                raise KeyError(document.id)
        document.updated_at = values['updated_at']
        return document

    def delete(self, id, *, application_id=None):
        with self.engine.begin() as connection:
            return bool(connection.execute(self._scope(self.table.delete().where(
                self.table.c.id == id), application_id)).rowcount)

    def list(self, limit=100, offset=0, *, application_id=None):
        positive_limit(limit)
        if type(offset) is not int or offset < 0:
            raise ValueError('Offset must be a nonnegative integer')
        with self.engine.connect() as connection:
            rows = connection.execute(self._scope(sa.select(self.table), application_id).order_by(
                self.table.c.id).limit(limit).offset(offset)).mappings().all()
        return [self._document(row) for row in rows]

    def search_by_vector(self, embedding, limit=10, filters=None, *, application_id):
        if not isinstance(application_id, UUID):
            raise ValueError('Search requires an application ID')
        vector = validate_embedding(embedding, len(embedding) if isinstance(embedding, (list, tuple)) else 0)
        positive_limit(limit)
        distance = self.table.c.embedding.cosine_distance(vector).label('distance')
        query = sa.select(self.table, distance).where(self.table.c.application_id == application_id)
        if filters is not None:
            if not isinstance(filters, dict):
                raise ValueError('Filters must be scalar equality mappings')
            for key, value in filters.items():
                if (key not in self.table.c or key in ('embedding', 'metadata')
                        or isinstance(value, (dict, list, tuple, set))):
                    raise ValueError('Only scalar field equality filters are supported')
                query = query.where(self.table.c[key] == value)
        application = PostgreSQLApplicationRepository(self.engine, self.applications).get(application_id)
        if application is None:
            raise KeyError(application_id)
        validate_embedding(vector, application.embedding_size)
        query = query.order_by(distance, self.table.c.id).limit(limit)
        with self.engine.connect() as connection:
            rows = connection.execute(query).mappings().all()
        matches = []
        for row in rows:
            values = dict(row)
            score = values.pop('distance')
            if score is None or not math.isfinite(score):
                raise ValueError('Stored embedding has undefined cosine distance')
            matches.append(SearchResult(self._document(values), float(score)))
        return matches

    def close(self) -> None:
        """Release the pool when this repository was obtained as an owned resource."""
        self.engine.dispose()


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
