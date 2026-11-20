"""PostgreSQL database lifecycle and repository ownership."""
import sqlalchemy as sa

from app.exceptions import DatabaseInitializationError, DatabaseResetError
from app.database.errors import database_errors
from app.resources import close_resource
from app.database.base import DatabaseBackend
from app.database.repositories.application import ApplicationRepository
from app.database.repositories.document import DocumentRepository
from app.database.repositories.processing_job import ProcessingJobRepository
from app.models.processing_job import ProcessingJob
from app.database.config import validate_config
from app.database.postgresql_connection import create_database, database_url, drop_database
from app.database.schema import build_metadata, initialize_schema


class PostgreSQLBackend(DatabaseBackend):
    def __init__(self, settings):
        self.settings = validate_config(settings)
        self.url = database_url(self.settings)
        with database_errors():
            self.engine = sa.create_engine(self.url, pool_pre_ping=True)
        self.metadata = build_metadata(self.settings['dimensions'])

    @property
    def maintenance_url(self):
        return self.url.set(database=self.settings['maintenance_database'])

    def create_database(self):
        result = create_database(self.maintenance_url, self.url.database)
        try:
            result.update(self.migrate())
        except Exception:
            if result['created']:
                raise DatabaseInitializationError(
                    'Database created, but schema initialization failed. Check pgvector '
                    'availability and permissions, then retry db:migrate.',
                    operation='db:create', completed_steps=('create_database',),
                ) from None
            raise
        return result

    @database_errors()
    def migrate(self):
        return initialize_schema(self.engine, self.metadata)

    def drop_database(self):
        self.close()
        return drop_database(self.maintenance_url, self.url.database)

    def reset(self):
        result = self.drop_database()
        try:
            result.update(self.create_database())
        except Exception:
            raise DatabaseResetError(
                'db:reset completed the drop step, but recreation failed. '
                'Previous database data is gone; check configuration, pgvector and permissions, '
                'then retry db:create.',
                operation='db:reset', completed_steps=('drop',),
            ) from None
        return result

    def documents(self):
        from app.database.repositories.postgresql.document import PostgreSQLDocumentRepository
        return PostgreSQLDocumentRepository(self.engine, self.metadata.tables['public.documents'])

    def embeddings(self):
        from app.database.repositories.postgresql.embedding import PostgreSQLEmbeddingRepository
        return PostgreSQLEmbeddingRepository(self.engine, self.metadata.tables['public.embeddings'])

    def applications(self) -> ApplicationRepository:
        from app.database.repositories.postgresql.application import PostgreSQLApplicationRepository
        return PostgreSQLApplicationRepository(self.engine, self.metadata.tables['public.applications'])

    def processing_jobs(self) -> ProcessingJobRepository:
        from app.database.repositories.postgresql.processing_job import PostgreSQLProcessingJobRepository
        return PostgreSQLProcessingJobRepository(self.engine, self.metadata.tables['public.document_processing_jobs'])

    def processing_documents(self, job: ProcessingJob) -> DocumentRepository:
        from app.database.repositories.postgresql.document import PostgreSQLDocumentRepository
        return PostgreSQLDocumentRepository(self.engine, self.metadata.tables['public.documents'], processing_claim=job)

    def close(self):
        close_resource(self.engine.dispose)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        close_resource(self.close, exc)
