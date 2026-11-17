import sqlalchemy as sa
from sqlalchemy.exc import DBAPIError

from app.database.config import validate_database_target


def create_database(maintenance_url, database_name):
    validate_database_target(database_name, maintenance_url.database)
    engine = sa.create_engine(maintenance_url)
    try:
        with engine.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        ) as connection:
            present = connection.execute(
                sa.text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": database_name},
            ).scalar()
            if present:
                return {"created": False}
            quoted = connection.dialect.identifier_preparer.quote(
                database_name
            )
            try:
                connection.exec_driver_sql(f"CREATE DATABASE {quoted}")
            except DBAPIError as exc:
                # PostgreSQL duplicate_database; another creator may win.
                if getattr(exc.orig, "sqlstate", None) != "42P04":
                    raise
                return {"created": False}
            return {"created": True}
    finally:
        engine.dispose()

def drop_database(maintenance_url, database_name):
    validate_database_target(database_name, maintenance_url.database)
    engine = sa.create_engine(maintenance_url)
    try:
        with engine.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        ) as connection:
            present = connection.execute(
                sa.text("SELECT 1 FROM pg_database WHERE datname = :name"),
                {"name": database_name},
            ).scalar()
            if not present:
                return {"dropped": False}
            quoted = connection.dialect.identifier_preparer.quote(
                database_name
            )
            try:
                connection.exec_driver_sql(f"DROP DATABASE {quoted}")
            except DBAPIError as exc:
                # invalid_catalog_name: a concurrent drop may have won.
                if getattr(exc.orig, "sqlstate", None) != "3D000":
                    raise
                return {"dropped": False}
            return {"dropped": True}
    finally:
        engine.dispose()


def database_url(settings):
    return sa.engine.URL.create(
        'postgresql+psycopg', username=settings['user'],
        password=settings.get('password'), host=settings['host'],
        port=int(settings.get('port', 5432)), database=settings['database'],
    )
