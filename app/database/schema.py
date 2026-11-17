"""Explicit application schema. Adding a table is a deliberate code change."""
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID


def build_metadata(dimensions=768):
    if type(dimensions) is not int or not 1 <= dimensions <= 16000:
        raise ValueError('dimensions must be an integer between 1 and 16000')
    metadata = sa.MetaData(schema='public', naming_convention={'pk': 'pk_%(table_name)s'})
    sa.Table(
        'documents', metadata,
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(dimensions), nullable=False),
        sa.Column('storage_key', sa.Text(), nullable=True),
        sa.Column('filename', sa.Text(), nullable=False),
        sa.Column('content_type', sa.Text(), nullable=False),
        sa.Column('source', sa.Text(), nullable=False),
        sa.Column('source_id', UUID(as_uuid=True), nullable=True),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.Text(), nullable=False),
        sa.Column('metadata', JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
    )
    return metadata


def initialize_schema(engine, metadata):
    """Validate existing tables before creating missing ones, in one transaction."""
    with engine.begin() as connection:
        connection.exec_driver_sql('CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public')
        inspector = sa.inspect(connection)
        existing = set(inspector.get_table_names(schema='public'))
        missing = []
        for table in metadata.sorted_tables:
            if table.name not in existing:
                missing.append(table.name)
                continue
            columns = {column['name']: column for column in inspector.get_columns(table.name, schema='public')}
            incompatible = set(columns) != set(table.c.keys())
            for column in table.c:
                actual = columns.get(column.name)
                if actual is None:
                    continue
                expected_type = str(column.type.compile(dialect=connection.dialect))
                actual_type = str(actual['type'].compile(dialect=connection.dialect))
                incompatible |= actual_type != expected_type or actual['nullable'] != column.nullable
            primary_key = inspector.get_pk_constraint(table.name, schema='public')['constrained_columns']
            incompatible |= primary_key != [column.name for column in table.primary_key]
            if incompatible:
                raise ValueError(
                    f'Incompatible schema for {table.name}; no tables were changed. '
                    'Apply a deliberate schema change, or use db:reset to discard database data.'
                )
        metadata.create_all(connection)
    return {'tables_created': missing}
