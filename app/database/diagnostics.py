"""Read-only PostgreSQL readiness checks; never run schema initialization."""
import sqlalchemy as sa

from app.database.postgresql_connection import database_url
from app.database.schema import build_metadata
from app.models.diagnostics import ConfigurationCheck


def _table_check(connection, dimensions: int) -> ConfigurationCheck:
    inspector = sa.inspect(connection)
    metadata = build_metadata(dimensions)
    existing = set(inspector.get_table_names(schema='public'))
    missing = [table.name for table in metadata.sorted_tables if table.name not in existing]
    if missing:
        return ConfigurationCheck('Database tables', 'REQUIRED',
                                  f'Missing public tables: {", ".join(missing)}. Run rypiq db:migrate.')
    missing_columns = []
    for table in metadata.sorted_tables:
        columns = {column['name'] for column in inspector.get_columns(table.name, schema='public')}
        missing_columns.extend(f'{table.name}.{column.name}' for column in table.columns if column.name not in columns)
    if missing_columns:
        return ConfigurationCheck('Database tables', 'INVALID',
                                  f'Missing columns: {", ".join(missing_columns)}. Run rypiq db:migrate; '
                                  'incompatible schemas require a deliberate schema change.')
    return ConfigurationCheck('Database tables', 'COMPLETE',
                              'Required public tables and columns exist; types, constraints, and write permissions were not verified.')


def check_database(*, settings: dict) -> list[ConfigurationCheck]:
    """Connect only to the selected database with read-only, time-limited queries."""
    engine = None
    results = []
    try:
        engine = sa.create_engine(database_url(settings), connect_args={
            'connect_timeout': 5,
            'options': '-c default_transaction_read_only=on -c statement_timeout=5000',
        })
        with engine.connect() as connection:
            results.append(ConfigurationCheck('Database connection', 'COMPLETE', 'Connected to the selected PostgreSQL database.'))
            vector = connection.execute(sa.text(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector')"
            )).scalar_one()
            results.append(ConfigurationCheck('pgvector extension', 'COMPLETE' if vector else 'REQUIRED',
                'pgvector is installed in the selected database.' if vector else
                'Install pgvector on the server, then run rypiq db:migrate with extension permissions.'))
            results.append(_table_check(connection, settings.get('dimensions', 768)))
    except sa.exc.SQLAlchemyError as error:
        state = getattr(getattr(error, 'orig', None), 'sqlstate', None)
        if not results:
            if state == '3D000':
                detail = 'Selected database does not exist. Run rypiq db:create.'
            elif state in ('28P01', '28000'):
                detail = 'Authentication failed; check the configured user and credentials.'
            else:
                detail = 'Cannot connect; check host, port, credentials, database existence, and server availability (5-second connection timeout).'
            results.append(ConfigurationCheck('Database connection', 'INVALID', detail))
        else:
            results.append(ConfigurationCheck('Database inspection', 'INVALID',
                           'Read-only inspection failed; check permissions and server availability. Each query has a 5-second timeout.'))
    finally:
        if engine is not None:
            engine.dispose()
    return results
