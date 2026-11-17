import os
from uuid import uuid4

import pytest
import sqlalchemy as sa

from app.database.backends.postgresql import PostgreSQLBackend


@pytest.fixture
def backend():
    raw = os.getenv('RYPIQ_TEST_ADMIN_URL')
    if not raw:
        pytest.skip('Set RYPIQ_TEST_ADMIN_URL to a disposable PostgreSQL server with pgvector')
    url = sa.engine.make_url(raw)
    if url.drivername != 'postgresql+psycopg':
        pytest.fail('RYPIQ_TEST_ADMIN_URL must use postgresql+psycopg')
    settings = dict(adapter='postgresql', host=url.host, port=url.port or 5432,
                    user=url.username, password=url.password,
                    database='rypiq_test_' + uuid4().hex,
                    maintenance_database=url.database, dimensions=3)
    backend = PostgreSQLBackend(settings)
    yield backend
    backend.close()
    admin = sa.create_engine(url, isolation_level='AUTOCOMMIT')
    try:
        with admin.connect() as connection:
            quoted = connection.dialect.identifier_preparer.quote(settings['database'])
            connection.exec_driver_sql(f'DROP DATABASE IF EXISTS {quoted}')
    finally:
        admin.dispose()
