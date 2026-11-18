"""PostgreSQL database lifecycle and repository ownership."""
import sqlalchemy as sa

from app.database.base import DatabaseBackend
from app.database.repositories.base import ApplicationRepository
from app.database.config import validate_config
from app.database.postgresql_connection import create_database, database_url, drop_database
from app.database.schema import build_metadata, initialize_schema


class PostgreSQLBackend(DatabaseBackend):
    def __init__(self, settings):
        self.settings = validate_config(settings)
        self.url = database_url(self.settings)
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
                raise RuntimeError(
                    'Database created, but schema initialization failed. Check pgvector '
                    'availability and permissions, then retry db:migrate.'
                ) from None
            raise
        return result

    def migrate(self):
        try:
            return initialize_schema(self.engine, self.metadata)
        except sa.exc.DBAPIError as exc:
            if getattr(exc.orig, 'sqlstate', None) in (None, '3D000'):
                raise ValueError('Cannot connect to database; check connectivity and credentials. '
                                 'If it does not exist, run db:create first.') from None
            raise

    def drop_database(self):
        self.close()
        return drop_database(self.maintenance_url, self.url.database)

    def reset(self):
        result = self.drop_database()
        try:
            result.update(self.create_database())
        except Exception:
            raise RuntimeError(
                'db:reset completed the drop step, but recreation failed. '
                'Previous database data is gone; check configuration, pgvector and permissions, '
                'then retry db:create.'
            ) from None
        return result

    def documents(self):
        from app.database.repositories.postgresql import PostgreSQLDocumentRepository
        return PostgreSQLDocumentRepository(self.engine, self.metadata.tables['public.documents'])

    def embeddings(self):
        from app.database.repositories.postgresql import PostgreSQLEmbeddingRepository
        return PostgreSQLEmbeddingRepository(self.engine, self.metadata.tables['public.embeddings'])

    def applications(self) -> ApplicationRepository:
        from app.database.repositories.postgresql import PostgreSQLApplicationRepository
        return PostgreSQLApplicationRepository(self.engine, self.metadata.tables['public.applications'])

    def close(self):
        self.engine.dispose()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
