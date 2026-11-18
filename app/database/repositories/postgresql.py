"""SQLAlchemy Core persistence and application-scoped cosine search."""
from uuid import UUID
import math
from dataclasses import asdict
from datetime import datetime, timezone

import sqlalchemy as sa

from app.database.repositories.base import DocumentRepository, EmbeddingRepository, ApplicationRepository
from app.models.application import Application, validate_application
from app.models.embedding import Embedding, validate_embedding
from app.models.document import Document, DocumentDetails
from app.models.search import SearchResult
from app.models.validation import validate_document, validate_chunk, validate_metadata
from app.providers.config import embedding_preprocessing


def positive_limit(limit):
    if type(limit) is not int or limit < 1:
        raise ValueError('Limit must be a positive integer')


class PostgreSQLDocumentRepository(DocumentRepository):
    def __init__(self, engine, table):
        self.engine = engine
        self.table = table
        self.applications = table.metadata.tables['public.applications']
        self.embeddings = table.metadata.tables['public.embeddings']

    def _application(self, connection, application_id):
        row = connection.execute(sa.select(self.applications).where(
            self.applications.c.id == application_id).with_for_update(read=True)).mappings().first()
        if row is None:
            raise KeyError(application_id)
        return row

    def create(self, document):
        validate_document(document)
        with self.engine.begin() as connection:
            self._application(connection, document.application_id)
            connection.execute(self.table.insert().values(**asdict(document)))
        return document

    def create_with_embeddings(self, document: Document, embeddings: list[Embedding]) -> DocumentDetails:
        validate_document(document)
        values = []
        for item in embeddings:
            vector = validate_chunk(item)
            if item.document_id != document.id:
                raise ValueError('Embedding belongs to another document')
            values.append({**asdict(item), 'embedding': vector})
        with self.engine.begin() as connection:
            application = self._application(connection, document.application_id)
            for item in embeddings:
                PostgreSQLEmbeddingRepository.validate_snapshot(item, application)
            connection.execute(self.table.insert().values(**asdict(document)))
            if values:
                connection.execute(self.embeddings.insert(), values)
        return DocumentDetails(document, embeddings)

    def _scope(self, statement, application_id):
        if application_id is not None:
            statement = statement.where(self.table.c.application_id == application_id)
        return statement

    def get(self, id, *, application_id=None):
        with self.engine.connect() as connection:
            row = connection.execute(self._scope(sa.select(self.table).where(
                self.table.c.id == id), application_id)).mappings().first()
        return Document(**row) if row else None

    def details(self, id: UUID, *, application_id: UUID | None = None) -> DocumentDetails | None:
        # A single joined statement gives a consistent parent/children snapshot.
        query = self._scope(sa.select(self.table, *[
            column.label('chunk_' + column.name) for column in self.embeddings.c
        ]).select_from(self.table.outerjoin(self.embeddings)).where(self.table.c.id == id), application_id)
        with self.engine.connect() as connection:
            rows = connection.execute(query.order_by(self.embeddings.c.chunk_index,
                                                      self.embeddings.c.id)).mappings().all()
        if not rows:
            return None
        parent = Document(**{c.name: rows[0][c.name] for c in self.table.c})
        chunks = [PostgreSQLEmbeddingRepository.from_row({c.name: row['chunk_' + c.name]
                  for c in self.embeddings.c}) for row in rows if row['chunk_id'] is not None]
        return DocumentDetails(parent, chunks)

    def update(self, document, *, application_id=None):
        validate_document(document)
        if application_id is not None and document.application_id != application_id:
            raise ValueError('Document does not match application scope')
        values = asdict(document)
        values['updated_at'] = datetime.now(timezone.utc)
        with self.engine.begin() as connection:
            self._application(connection, document.application_id)
            original = connection.execute(self._scope(sa.select(self.table).where(
                self.table.c.id == document.id), application_id).with_for_update()).mappings().first()
            if original is None:
                raise KeyError(document.id)
            if original['application_id'] != document.application_id:
                raise ValueError('Document ownership cannot be changed')
            connection.execute(self.table.update().where(self.table.c.id == document.id).values(**values))
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
        return [Document(**row) for row in rows]

    def close(self):
        self.engine.dispose()


class PostgreSQLEmbeddingRepository(EmbeddingRepository):
    def __init__(self, engine, table):
        self.engine = engine
        self.table = table
        self.documents = table.metadata.tables['public.documents']
        self.applications = table.metadata.tables['public.applications']

    @staticmethod
    def from_row(row):
        values = dict(row)
        values['embedding'] = [float(value) for value in values['embedding']]
        return Embedding(**values)

    @staticmethod
    def validate_snapshot(embedding, application):
        for name in ('embedding_model', 'embedding_size', 'provision_type'):
            if getattr(embedding, name) != application[name]:
                raise ValueError('Embedding settings must match its application')

    def _scope(self, statement, application_id):
        if application_id is not None:
            statement = statement.where(self.table.c.document_id.in_(
                sa.select(self.documents.c.id).where(self.documents.c.application_id == application_id)))
        return statement

    def _validate_parent(self, connection, item, application_id):
        # Lock the application before the parent, matching document persistence.
        owner = sa.select(self.documents.c.application_id).where(self.documents.c.id == item.document_id).scalar_subquery()
        query = sa.select(self.applications).where(self.applications.c.id == owner)
        if application_id is not None:
            query = query.where(self.applications.c.id == application_id)
        application = connection.execute(query.with_for_update(read=True)).mappings().first()
        if application is None:
            raise KeyError(item.document_id)
        parent = connection.execute(sa.select(self.documents.c.id).where(
            self.documents.c.id == item.document_id,
            self.documents.c.application_id == application['id']).with_for_update(read=True)).first()
        if parent is None:
            raise KeyError(item.document_id)
        self.validate_snapshot(item, application)

    def create(self, embedding: Embedding, *, application_id: UUID | None = None) -> Embedding:
        values = {**asdict(embedding), 'embedding': validate_chunk(embedding)}
        with self.engine.begin() as connection:
            self._validate_parent(connection, embedding, application_id)
            connection.execute(self.table.insert().values(**values))
        return embedding

    def get(self, id: UUID, *, application_id: UUID | None = None) -> Embedding | None:
        with self.engine.connect() as connection:
            row = connection.execute(self._scope(sa.select(self.table).where(
                self.table.c.id == id), application_id)).mappings().first()
        return self.from_row(row) if row else None

    def update(self, embedding: Embedding, *, application_id: UUID | None = None) -> Embedding:
        values = {**asdict(embedding), 'embedding': validate_chunk(embedding),
                  'updated_at': datetime.now(timezone.utc)}
        with self.engine.begin() as connection:
            self._validate_parent(connection, embedding, application_id)
            original = connection.execute(self._scope(sa.select(self.table).where(
                self.table.c.id == embedding.id), application_id).with_for_update()).mappings().first()
            if original is None:
                raise KeyError(embedding.id)
            if any(values[key] != original[key] for key in (
                    'document_id', 'embedding_model', 'embedding_size', 'provision_type')):
                raise ValueError('Embedding ownership and model snapshots cannot be changed')
            connection.execute(self.table.update().where(self.table.c.id == embedding.id).values(**values))
        embedding.updated_at = values['updated_at']
        return embedding

    def delete(self, id: UUID, *, application_id: UUID | None = None) -> bool:
        with self.engine.begin() as connection:
            return bool(connection.execute(self._scope(self.table.delete().where(
                self.table.c.id == id), application_id)).rowcount)

    def list(self, *, document_id: UUID, application_id: UUID | None = None) -> list[Embedding]:
        with self.engine.connect() as connection:
            rows = connection.execute(self._scope(sa.select(self.table).where(
                self.table.c.document_id == document_id), application_id).order_by(
                    self.table.c.chunk_index, self.table.c.id)).mappings().all()
        return [self.from_row(row) for row in rows]

    def search_by_vector(self, embedding: list[float], limit: int = 10, filters: dict | None = None, *,
                         application_id: UUID, metadata_filters: dict | None = None) -> list[SearchResult]:
        if not isinstance(application_id, UUID):
            raise ValueError('Search requires an application ID')
        vector = validate_embedding(embedding, len(embedding) if isinstance(embedding, (list, tuple)) else 0)
        positive_limit(limit)
        distance = self.table.c.embedding.cosine_distance(vector).label('distance')
        query = sa.select(self.table, *[column.label('parent_' + column.name) for column in self.documents.c],
                          distance).select_from(self.table.join(self.documents)).where(
                              self.documents.c.application_id == application_id)
        if metadata_filters is not None:
            validate_metadata(metadata_filters)
            query = query.where(self.table.c.metadata.contains(metadata_filters))
        if filters is not None:
            if not isinstance(filters, dict):
                raise ValueError('Filters must be scalar equality mappings')
            columns = {c.name: c for c in self.documents.c if c.name not in ('id', 'metadata')}
            columns.update({c.name: c for c in self.table.c if c.name not in ('embedding', 'metadata')})
            for key, value in filters.items():
                if key not in columns or isinstance(value, (dict, list, tuple, set)):
                    raise ValueError('Only scalar field equality filters are supported')
                query = query.where(columns[key] == value)
        with self.engine.connect() as connection:
            application = connection.execute(sa.select(self.applications).where(
                self.applications.c.id == application_id)).mappings().first()
            if application is None:
                raise KeyError(application_id)
            validate_embedding(vector, application['embedding_size'])
            rows = connection.execute(query.order_by(distance, self.table.c.id).limit(limit)).mappings().all()
        matches = []
        for row in rows:
            score = row['distance']
            if score is None or not math.isfinite(score):
                raise ValueError('Stored embedding has undefined cosine distance')
            chunk = self.from_row({c.name: row[c.name] for c in self.table.c})
            document = Document(**{c.name: row['parent_' + c.name] for c in self.documents.c})
            matches.append(SearchResult(chunk, document, float(score)))
        return matches

    def close(self):
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
