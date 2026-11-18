from uuid import UUID
import math
from dataclasses import asdict
from datetime import datetime, timezone

import sqlalchemy as sa

from app.database.errors import database_connection
from app.resources import close_resource
from app.exceptions import (
    ApplicationNotFoundError,
    DocumentNotFoundError,
    EmbeddingNotFoundError,
    ImmutableFieldError,
    InvalidEmbeddingError,
    InvalidIdentifierError,
    ModelMismatchError,
    ValidationError,
)
from app.database.repositories.embedding import EmbeddingRepository
from app.models.validation import validate_limit
from app.models.document import Document
from app.models.embedding import Embedding, validate_embedding
from app.models.search import SearchResult
from app.models.validation import validate_chunk, validate_metadata


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
                raise ModelMismatchError('Embedding settings must match its application')

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
            raise DocumentNotFoundError(item.document_id, resource_id=item.document_id)
        parent = connection.execute(sa.select(self.documents.c.id).where(
            self.documents.c.id == item.document_id,
            self.documents.c.application_id == application['id']).with_for_update(read=True)).first()
        if parent is None:
            raise DocumentNotFoundError(item.document_id, resource_id=item.document_id)
        self.validate_snapshot(item, application)

    def create(self, embedding: Embedding, *, application_id: UUID | None = None) -> Embedding:
        values = {**asdict(embedding), 'embedding': validate_chunk(embedding)}
        with database_connection(self.engine, write=True) as connection:
            self._validate_parent(connection, embedding, application_id)
            connection.execute(self.table.insert().values(**values))
        return embedding

    def get(self, id: UUID, *, application_id: UUID | None = None) -> Embedding | None:
        with database_connection(self.engine) as connection:
            row = connection.execute(self._scope(sa.select(self.table).where(
                self.table.c.id == id), application_id)).mappings().first()
        return self.from_row(row) if row else None

    def update(self, embedding: Embedding, *, application_id: UUID | None = None) -> Embedding:
        values = {**asdict(embedding), 'embedding': validate_chunk(embedding),
                  'updated_at': datetime.now(timezone.utc)}
        with database_connection(self.engine, write=True) as connection:
            self._validate_parent(connection, embedding, application_id)
            original = connection.execute(self._scope(sa.select(self.table).where(
                self.table.c.id == embedding.id), application_id).with_for_update()).mappings().first()
            if original is None:
                raise EmbeddingNotFoundError(embedding.id, resource_id=embedding.id)
            if any(values[key] != original[key] for key in (
                    'document_id', 'embedding_model', 'embedding_size', 'provision_type')):
                raise ImmutableFieldError('Embedding ownership and model snapshots cannot be changed')
            connection.execute(self.table.update().where(self.table.c.id == embedding.id).values(**values))
        embedding.updated_at = values['updated_at']
        return embedding

    def delete(self, id: UUID, *, application_id: UUID | None = None) -> bool:
        with database_connection(self.engine, write=True) as connection:
            return bool(connection.execute(self._scope(self.table.delete().where(
                self.table.c.id == id), application_id)).rowcount)

    def list(self, *, document_id: UUID, application_id: UUID | None = None) -> list[Embedding]:
        with database_connection(self.engine) as connection:
            rows = connection.execute(self._scope(sa.select(self.table).where(
                self.table.c.document_id == document_id), application_id).order_by(
                    self.table.c.chunk_index, self.table.c.id)).mappings().all()
        return [self.from_row(row) for row in rows]

    def search_by_vector(self, embedding: list[float], limit: int = 10, filters: dict | None = None, *,
                         application_id: UUID, metadata_filters: dict | None = None) -> list[SearchResult]:
        if not isinstance(application_id, UUID):
            raise InvalidIdentifierError('Search requires an application ID')
        vector = validate_embedding(embedding, len(embedding) if isinstance(embedding, (list, tuple)) else 0)
        validate_limit(limit)
        distance = self.table.c.embedding.cosine_distance(vector).label('distance')
        query = sa.select(self.table, *[column.label('parent_' + column.name) for column in self.documents.c],
                          distance).select_from(self.table.join(self.documents)).where(
                              self.documents.c.application_id == application_id)
        if metadata_filters is not None:
            validate_metadata(metadata_filters)
            query = query.where(self.table.c.metadata.contains(metadata_filters))
        if filters is not None:
            if not isinstance(filters, dict):
                raise ValidationError('Filters must be scalar equality mappings')
            columns = {c.name: c for c in self.documents.c if c.name not in ('id', 'metadata')}
            columns.update({c.name: c for c in self.table.c if c.name not in ('embedding', 'metadata')})
            for key, value in filters.items():
                if key not in columns or isinstance(value, (dict, list, tuple, set)):
                    raise ValidationError('Only scalar field equality filters are supported')
                query = query.where(columns[key] == value)
        with database_connection(self.engine) as connection:
            application = connection.execute(sa.select(self.applications).where(
                self.applications.c.id == application_id)).mappings().first()
            if application is None:
                raise ApplicationNotFoundError(application_id, resource_id=application_id)
            validate_embedding(vector, application['embedding_size'])
            rows = connection.execute(query.order_by(distance, self.table.c.id).limit(limit)).mappings().all()
        matches = []
        for row in rows:
            score = row['distance']
            if score is None or not math.isfinite(score):
                raise InvalidEmbeddingError('Stored embedding has undefined cosine distance')
            chunk = self.from_row({c.name: row[c.name] for c in self.table.c})
            document = Document(**{c.name: row['parent_' + c.name] for c in self.documents.c})
            matches.append(SearchResult(chunk, document, float(score)))
        return matches

    def close(self):
        close_resource(self.engine.dispose)
