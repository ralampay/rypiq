from uuid import UUID
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone

import sqlalchemy as sa

from app.database.repositories.document import DocumentRepository
from app.database.repositories.postgresql._validation import positive_limit
from app.database.repositories.postgresql.embedding import PostgreSQLEmbeddingRepository
from app.models.document import Document, DocumentDetails, DocumentSummary
from app.models.embedding import Embedding
from app.models.validation import validate_document, validate_chunk, normalize_document_name


class PostgreSQLDocumentRepository(DocumentRepository):
    def __init__(self, engine, table):
        self.engine = engine
        self.table = table
        self.applications = table.metadata.tables['public.applications']
        self.embeddings = table.metadata.tables['public.embeddings']

    @contextmanager
    def _write(self):
        try:
            with self.engine.begin() as connection:
                yield connection
        except sa.exc.IntegrityError as error:
            if (getattr(error.orig, 'sqlstate', None) == '23505'
                    and getattr(getattr(error.orig, 'diag', None), 'constraint_name', None)
                    == 'uq_documents_application_name'):
                raise ValueError('A document with this name already exists in the application') from None
            raise

    def name_exists(self, *, application_id: UUID, name: str) -> bool:
        if not isinstance(application_id, UUID):
            raise ValueError('Document lookup requires an application UUID')
        name = normalize_document_name(name)
        with self.engine.connect() as connection:
            return connection.scalar(sa.select(sa.exists().where(
                self.table.c.application_id == application_id, self.table.c.name == name)))

    def list_summaries(self, *, application_id: UUID, limit: int = 100,
                       offset: int = 0) -> list[DocumentSummary]:
        if not isinstance(application_id, UUID):
            raise ValueError('Document listing requires an application UUID')
        positive_limit(limit)
        if type(offset) is not int or offset < 0:
            raise ValueError('Offset must be a nonnegative integer')
        query = sa.select(self.table.c.id, self.table.c.name,
                          sa.func.count(self.embeddings.c.id).label('embedding_count')).select_from(
                              self.table.outerjoin(self.embeddings)).where(
                                  self.table.c.application_id == application_id).group_by(
                                      self.table.c.id).order_by(self.table.c.id).limit(limit).offset(offset)
        with self.engine.connect() as connection:
            rows = connection.execute(query).mappings().all()
        return [DocumentSummary(**row) for row in rows]

    def _application(self, connection, application_id):
        row = connection.execute(sa.select(self.applications).where(
            self.applications.c.id == application_id).with_for_update(read=True)).mappings().first()
        if row is None:
            raise KeyError(application_id)
        return row

    def create(self, document):
        validate_document(document)
        with self._write() as connection:
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
        with self._write() as connection:
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
        with self._write() as connection:
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
        with self._write() as connection:
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
