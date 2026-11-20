"""PostgreSQL queue with expiring, fenced claims and short transactions."""
from uuid import UUID, uuid4

import sqlalchemy as sa

from app.database.errors import database_connection
from app.exceptions import ConflictError, DocumentNotFoundError, ValidationError
from app.models.processing_job import ProcessingJob
from app.models.validation import validate_id

ACTIVE = ('queued', 'running')
LEASE = sa.text("INTERVAL '60 seconds'")


def claim_predicate(table, job):
    return sa.and_(table.c.id == job.id, table.c.state == 'running',
                   table.c.claim_token == job.claim_token,
                   table.c.lease_expires_at > sa.func.clock_timestamp())


def lock_claim(connection, table, job):
    # Serialize completion/failure with reclaim. Recheck wall-clock expiry after lock acquisition.
    row = connection.execute(sa.select(table.c.id).where(
        claim_predicate(table, job)).with_for_update()).first()
    if row is None or not connection.scalar(sa.select(sa.exists().where(claim_predicate(table, job)))):
        raise ConflictError('Processing claim expired; this worker cannot write results')


def complete_claim(connection, table, job):
    updated = connection.execute(table.update().where(claim_predicate(table, job)).values(
        state='succeeded', lease_expires_at=None, error=None, updated_at=sa.func.clock_timestamp()))
    if not updated.rowcount:
        raise ConflictError('Processing claim expired; this worker cannot write results')


class PostgreSQLProcessingJobRepository:
    def __init__(self, engine, table):
        self.engine, self.table = engine, table
        self.documents = table.metadata.tables['public.documents']

    def enqueue(self, *, document_id: UUID, batch_size: int = 1) -> ProcessingJob:
        validate_id(document_id)
        if type(batch_size) is not int or batch_size < 1:
            raise ValidationError('batch_size must be a positive integer')
        with database_connection(self.engine, write=True) as connection:
            document = connection.execute(sa.select(self.documents).where(
                self.documents.c.id == document_id).with_for_update()).mappings().first()
            if document is None:
                raise DocumentNotFoundError(document_id, resource_id=document_id)
            active = connection.execute(sa.select(self.table).where(
                self.table.c.document_id == document_id, self.table.c.state.in_(ACTIVE))).mappings().first()
            if active:
                return ProcessingJob(**active)
            if not document['storage_key']:
                raise ValidationError('Document has no stored original')
            row = connection.execute(self.table.insert().values(
                id=uuid4(), document_id=document_id, batch_size=batch_size, state='queued', attempts=0,
                created_at=sa.func.clock_timestamp(), updated_at=sa.func.clock_timestamp()
            ).returning(self.table)).mappings().one()
            return ProcessingJob(**row)

    def get(self, job_id: UUID) -> ProcessingJob | None:
        validate_id(job_id)
        with database_connection(self.engine) as connection:
            row = connection.execute(sa.select(self.table).where(self.table.c.id == job_id)).mappings().first()
        return ProcessingJob(**row) if row else None

    def latest(self, document_id: UUID) -> ProcessingJob | None:
        validate_id(document_id)
        with database_connection(self.engine) as connection:
            row = connection.execute(sa.select(self.table).where(self.table.c.document_id == document_id)
                .order_by(self.table.c.created_at.desc(), self.table.c.id.desc()).limit(1)).mappings().first()
        return ProcessingJob(**row) if row else None

    def claim(self) -> ProcessingJob | None:
        with database_connection(self.engine, write=True) as connection:
            eligible = sa.or_(self.table.c.state == 'queued', sa.and_(
                self.table.c.state == 'running', self.table.c.lease_expires_at <= sa.func.clock_timestamp()))
            row = connection.execute(sa.select(self.table).where(eligible)
                .order_by(self.table.c.updated_at, self.table.c.id).with_for_update(skip_locked=True)
                .limit(1)).mappings().first()
            if row is None:
                return None
            if row['attempts'] >= 3:
                connection.execute(self.table.update().where(self.table.c.id == row['id']).values(
                    state='failed', error='Processing interrupted too many times; submit a new job.',
                    lease_expires_at=None, updated_at=sa.func.clock_timestamp()))
                connection.execute(self.documents.update().where(self.documents.c.id == row['document_id']).values(
                    status='error', updated_at=sa.func.clock_timestamp()))
                return None
            row = connection.execute(self.table.update().where(self.table.c.id == row['id']).values(
                state='running', attempts=row['attempts'] + 1, claim_token=uuid4(), error=None,
                lease_expires_at=sa.func.clock_timestamp() + LEASE, updated_at=sa.func.clock_timestamp()
            ).returning(self.table)).mappings().one()
            return ProcessingJob(**row)

    def heartbeat(self, job: ProcessingJob) -> bool:
        with database_connection(self.engine, write=True) as connection:
            return bool(connection.execute(self.table.update().where(claim_predicate(self.table, job)).values(
                lease_expires_at=sa.func.clock_timestamp() + LEASE, updated_at=sa.func.clock_timestamp())).rowcount)

    def fail(self, job: ProcessingJob, *, message: str) -> bool:
        with database_connection(self.engine, write=True) as connection:
            try:
                lock_claim(connection, self.table, job)
            except ConflictError:
                return False
            connection.execute(self.table.update().where(self.table.c.id == job.id).values(
                state='failed', error=message, lease_expires_at=None, updated_at=sa.func.clock_timestamp()))
            connection.execute(self.documents.update().where(self.documents.c.id == job.document_id).values(
                status='error', updated_at=sa.func.clock_timestamp()))
            return True

    def release(self, job: ProcessingJob) -> bool:
        """A busy document (e.g. a CLI processor) is not a failed processing attempt."""
        with database_connection(self.engine, write=True) as connection:
            return bool(connection.execute(self.table.update().where(claim_predicate(self.table, job)).values(
                state='queued', claim_token=None, lease_expires_at=None, attempts=self.table.c.attempts - 1,
                updated_at=sa.func.clock_timestamp())).rowcount)
