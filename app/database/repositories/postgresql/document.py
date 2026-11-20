from uuid import UUID
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone

import sqlalchemy as sa

from app.database.errors import database_connection
from app.resources import close_resource
from app.exceptions import (
    ApplicationNotFoundError,
    DocumentNotFoundError,
    ImmutableFieldError,
    InvalidIdentifierError,
    ValidationError,
    ConflictError,
    ProcessingBusyError,
)
from app.database.repositories.document import DocumentRepository
from app.models.validation import validate_pagination, validate_status, validate_id
from app.database.repositories.postgresql.embedding import PostgreSQLEmbeddingRepository
from app.models.document import Document, DocumentDetails, DocumentSummary
from app.models.embedding import Embedding
from app.models.application import Application
from app.models.processing import SNAPSHOT_KEY, processing_settings
from app.models.validation import validate_document, validate_chunk, normalize_document_name


class PostgreSQLDocumentRepository(DocumentRepository):
    def __init__(self, engine, table, *, processing_claim=None):
        self.processing_claim = processing_claim
        self.jobs = table.metadata.tables.get("public.document_processing_jobs")
        self.engine = engine
        self.table = table
        self.applications = table.metadata.tables['public.applications']
        self.embeddings = table.metadata.tables['public.embeddings']

    @contextmanager
    def _write(self, document=None):
        with database_connection(
            self.engine, write=True,
            application_id=document.application_id if document is not None else None,
            name=document.name if document is not None else None,
        ) as connection:
            if self.processing_claim is not None:
                from app.database.repositories.postgresql.processing_job import lock_claim
                lock_claim(connection, self.jobs, self.processing_claim)
            yield connection

    def name_exists(self, *, application_id: UUID, name: str) -> bool:
        if not isinstance(application_id, UUID):
            raise InvalidIdentifierError('Document lookup requires an application UUID')
        name = normalize_document_name(name)
        with database_connection(self.engine) as connection:
            return connection.scalar(sa.select(sa.exists().where(
                self.table.c.application_id == application_id, self.table.c.name == name)))

    def list_summaries(self, *, application_id: UUID, limit: int = 100,
                       offset: int = 0, status: str | None = None) -> list[DocumentSummary]:
        if not isinstance(application_id, UUID):
            raise InvalidIdentifierError('Document listing requires an application UUID')
        validate_pagination(limit, offset)
        query = sa.select(self.table.c.id, self.table.c.name, self.table.c.status,
                          sa.func.count(self.embeddings.c.id).label('embedding_count')).select_from(
                              self.table.outerjoin(self.embeddings)).where(
                                  self.table.c.application_id == application_id).group_by(
                                      self.table.c.id).order_by(self.table.c.id).limit(limit).offset(offset)
        if status is not None:
            validate_status(status)
            query = query.where(self.table.c.status == status)
        with database_connection(self.engine) as connection:
            rows = connection.execute(query).mappings().all()
        return [DocumentSummary(**row) for row in rows]

    def _application(self, connection, application_id):
        row = connection.execute(sa.select(self.applications).where(
            self.applications.c.id == application_id).with_for_update(read=True)).mappings().first()
        if row is None:
            raise ApplicationNotFoundError(application_id, resource_id=application_id)
        return row

    def create(self, document):
        validate_document(document)
        with self._write(document) as connection:
            application = self._application(connection, document.application_id)
            document.metadata = {**document.metadata, SNAPSHOT_KEY: processing_settings(Application(**application))}
            connection.execute(self.table.insert().values(**asdict(document)))
        return document

    def create_with_embeddings(self, document: Document, embeddings: list[Embedding]) -> DocumentDetails:
        validate_document(document)
        values = []
        for item in embeddings:
            vector = validate_chunk(item)
            if item.document_id != document.id:
                raise ValidationError('Embedding belongs to another document')
            values.append({**asdict(item), 'embedding': vector})
        with self._write(document) as connection:
            application = self._application(connection, document.application_id)
            document.metadata = {**document.metadata, SNAPSHOT_KEY: processing_settings(Application(**application))}
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
        with database_connection(self.engine) as connection:
            row = connection.execute(self._scope(sa.select(self.table).where(
                self.table.c.id == id), application_id)).mappings().first()
        return Document(**row) if row else None

    def details(self, id: UUID, *, application_id: UUID | None = None) -> DocumentDetails | None:
        # A single joined statement gives a consistent parent/children snapshot.
        query = self._scope(sa.select(self.table, *[
            column.label('chunk_' + column.name) for column in self.embeddings.c
        ]).select_from(self.table.outerjoin(self.embeddings)).where(self.table.c.id == id), application_id)
        with database_connection(self.engine) as connection:
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
            raise ValidationError('Document does not match application scope')
        values = asdict(document)
        values['updated_at'] = datetime.now(timezone.utc)
        with self._write(document) as connection:
            self._application(connection, document.application_id)
            original = connection.execute(self._scope(sa.select(self.table).where(
                self.table.c.id == document.id), application_id).with_for_update()).mappings().first()
            if original is None:
                raise DocumentNotFoundError(document.id, resource_id=document.id)
            if original['application_id'] != document.application_id:
                raise ImmutableFieldError('Document ownership cannot be changed')
            if SNAPSHOT_KEY in original['metadata']:
                values['metadata'] = {**values['metadata'], SNAPSHOT_KEY: original['metadata'][SNAPSHOT_KEY]}
            connection.execute(self.table.update().where(self.table.c.id == document.id).values(**values))
        document.updated_at = values['updated_at']
        document.metadata = values['metadata']
        return document

    def delete(self, id, *, application_id=None):
        with self._write() as connection:
            original = connection.execute(self._scope(sa.select(self.table.c.id).where(
                self.table.c.id == id), application_id).with_for_update()).first()
            if original is None:
                return False
            if self.jobs is not None and connection.scalar(sa.select(sa.exists().where(
                    self.jobs.c.document_id == id, self.jobs.c.state.in_(['queued', 'running'])))):
                raise ConflictError('Document has active processing work')
            return bool(connection.execute(self._scope(self.table.delete().where(
                self.table.c.id == id), application_id)).rowcount)

    def list(self, limit=100, offset=0, *, application_id=None, status=None):
        validate_pagination(limit, offset)
        query = self._scope(sa.select(self.table), application_id)
        if status is not None:
            validate_status(status)
            query = query.where(self.table.c.status == status)
        with database_connection(self.engine) as connection:
            rows = connection.execute(query.order_by(self.table.c.id).limit(limit).offset(offset)).mappings().all()
        return [Document(**row) for row in rows]

    def close(self):
        close_resource(self.engine.dispose)

    def set_status(self, id: UUID, status: str) -> Document:
        validate_id(id)
        validate_status(status)
        with self._write() as connection:
            row = connection.execute(self.table.update().where(self.table.c.id == id).values(
                status=status, updated_at=datetime.now(timezone.utc)).returning(self.table)).mappings().first()
            if row is None:
                raise DocumentNotFoundError(id, resource_id=id)
        return Document(**row)

    @contextmanager
    def processing_lock(self, id: UUID):
        validate_id(id)
        # Session locks survive the short status/write transactions. A dedicated
        # connection owns the lock and must never return locked to the pool.
        key = int.from_bytes(id.bytes[:8], 'big', signed=True)
        with database_connection(self.engine) as connection:
            try:
                acquired = connection.scalar(sa.text('SELECT pg_try_advisory_lock(:key)'), {'key': key})
                connection.commit()
            except BaseException as failure:
                # A failed response may still have acquired the server-side lock.
                close_resource(connection.invalidate, failure)
                raise
            if not acquired:
                raise ProcessingBusyError('Document is already being processed')
            failure = None
            try:
                if self.processing_claim is None and self.jobs is not None:
                    active = connection.scalar(sa.select(sa.exists().where(
                        self.jobs.c.document_id == id, self.jobs.c.state.in_(['queued', 'running']))))
                    connection.commit()
                    if active:
                        raise ProcessingBusyError('Document has active queued processing work')
                yield
            except BaseException as exc:
                failure = exc
                raise
            finally:
                try:
                    connection.execute(sa.text('SELECT pg_advisory_unlock(:key)'), {'key': key})
                    connection.commit()
                except Exception as cleanup_failure:
                    close_resource(connection.invalidate, failure or cleanup_failure)
                    if failure is None:
                        raise
                    failure.add_note('Processing lock cleanup failed; its connection was discarded.')

    def replace_embeddings(self, document: Document, embeddings: list[Embedding], *, application: Application) -> DocumentDetails:
        validate_document(document)
        if application.id != document.application_id:
            raise ValidationError('Application does not match document ownership')
        values = []
        for item in embeddings:
            vector = validate_chunk(item)
            if item.document_id != document.id:
                raise ValidationError('Embedding belongs to another document')
            values.append({**asdict(item), 'embedding': vector})
        with self._write(document) as connection:
            current = self._application(connection, document.application_id)
            if processing_settings(Application(**current)) != processing_settings(application):
                raise ConflictError('Application processing settings changed; retry the document')
            original = connection.execute(sa.select(self.table).where(
                self.table.c.id == document.id).with_for_update()).mappings().first()
            if original is None:
                raise DocumentNotFoundError(document.id, resource_id=document.id)
            if original['application_id'] != document.application_id:
                raise ImmutableFieldError('Document ownership cannot be changed')
            for item in embeddings:
                PostgreSQLEmbeddingRepository.validate_snapshot(item, current)
            connection.execute(self.embeddings.delete().where(self.embeddings.c.document_id == document.id))
            if values:
                connection.execute(self.embeddings.insert(), values)
            row = connection.execute(self.table.update().where(self.table.c.id == document.id).values(
                status='done', updated_at=datetime.now(timezone.utc),
                metadata={**original['metadata'], SNAPSHOT_KEY: processing_settings(application)}
            ).returning(self.table)).mappings().one()
            if self.processing_claim is not None:
                from app.database.repositories.postgresql.processing_job import complete_claim
                complete_claim(connection, self.jobs, self.processing_claim)
        return DocumentDetails(Document(**row), embeddings)
