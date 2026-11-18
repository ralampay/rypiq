"""Explicit application schema. Adding a table is a deliberate code change."""
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID
from app.database.errors import database_connection
from app.exceptions import SchemaCompatibilityError, ValidationError


def build_metadata(dimensions=768):
    if type(dimensions) is not int or not 1 <= dimensions <= 16000:
        raise ValidationError('dimensions must be an integer between 1 and 16000')
    metadata = sa.MetaData(schema='public', naming_convention={'pk': 'pk_%(table_name)s', 'ix': 'ix_%(table_name)s_%(column_0_name)s'})
    sa.Table(
        'applications', metadata,
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('api_key', sa.Text(), nullable=False, unique=True),
        sa.Column('embedding_model', sa.Text(), nullable=False),
        sa.Column('provision_type', sa.Text(), nullable=False),
        sa.Column('embedding_size', sa.Integer(), nullable=False),
        sa.Column('inference_model', sa.Text(), nullable=True),
        sa.Column('data', JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("provision_type IN ('local', 'openai', 'llama-server')",
                           name='ck_applications_provider'),
        sa.CheckConstraint('embedding_size BETWEEN 1 AND 16000', name='ck_applications_size'),
    )
    sa.Table(
        'documents', metadata,
        sa.Column('name', sa.Text(), nullable=False),
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('application_id', UUID(as_uuid=True),
                  sa.ForeignKey('public.applications.id', ondelete='RESTRICT'), nullable=False, index=True),
        sa.Column('storage_key', sa.Text(), nullable=True),
        sa.Column('filename', sa.Text(), nullable=False),
        sa.Column('content_type', sa.Text(), nullable=False),
        sa.Column('source', sa.Text(), nullable=False),
        sa.Column('metadata', JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint('application_id', 'name', name='uq_documents_application_name'),
        sa.CheckConstraint("length(name) > 0", name='ck_documents_name_nonempty'),
    )
    embeddings = sa.Table(
        'embeddings', metadata,
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('document_id', UUID(as_uuid=True),
                  sa.ForeignKey('public.documents.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('model_name', sa.Text(), nullable=False),
        sa.Column('embedding_model', sa.Text(), nullable=False),
        sa.Column('embedding_size', sa.Integer(), nullable=False),
        sa.Column('provision_type', sa.Text(), nullable=False),
        sa.Column('metadata', JSONB(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint('vector_dims(embedding) = embedding_size', name='ck_embeddings_vector_size'),
        sa.CheckConstraint('embedding_size BETWEEN 1 AND 16000', name='ck_embeddings_size'),
        sa.CheckConstraint("provision_type IN ('local', 'openai', 'llama-server')", name='ck_embeddings_provider'),
    )
    sa.Index('ix_embeddings_metadata', embeddings.c.metadata, postgresql_using='gin')
    return metadata


def validate_table(connection, table):
    inspector = sa.inspect(connection)
    columns = {c['name']: c for c in inspector.get_columns(table.name, schema='public')}
    incompatible = set(columns) != set(table.c.keys())
    # Matching definitions alone do not guarantee that existing rows were checked.
    incompatible |= connection.execute(sa.text('''
        SELECT EXISTS (
            SELECT 1 FROM pg_constraint c
            JOIN pg_class t ON t.oid = c.conrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            WHERE n.nspname = 'public' AND t.relname = :table_name
              AND NOT c.convalidated
        )
    '''), {'table_name': table.name}).scalar_one()
    for column in table.c:
        actual = columns.get(column.name)
        if actual is None:
            continue
        incompatible |= (str(actual['type'].compile(dialect=connection.dialect)) !=
                         str(column.type.compile(dialect=connection.dialect)) or
                         actual['nullable'] != column.nullable)
    incompatible |= inspector.get_pk_constraint(table.name, schema='public')['constrained_columns'] != [
        c.name for c in table.primary_key]
    unique = {tuple(c['column_names']) for c in inspector.get_unique_constraints(table.name, schema='public')}
    for constraint in table.constraints:
        if isinstance(constraint, sa.UniqueConstraint):
            incompatible |= tuple(c.name for c in constraint.columns) not in unique
    foreign = inspector.get_foreign_keys(table.name, schema='public')
    for constraint in table.foreign_key_constraints:
        incompatible |= not any(
            item['constrained_columns'] == [c.name for c in constraint.columns]
            and item['referred_table'] == constraint.referred_table.name
            and item['referred_schema'] in (None, 'public')
            and item['referred_columns'] == [e.column.name for e in constraint.elements]
            and item.get('options', {}).get('ondelete') == constraint.ondelete
            for item in foreign)
    # Compare PostgreSQL-normalized check expressions against a temporary table
    # so formatting and PostgreSQL's IN-to-ANY rewrite cannot hide drift.
    checks = [c for c in table.constraints if isinstance(c, sa.CheckConstraint)]
    if checks:
        temporary = sa.Table('_rypiq_validate_' + table.name, sa.MetaData(),
                             *(sa.Column(c.name, c.type) for c in table.c),
                             *(sa.CheckConstraint(c.sqltext, name=c.name) for c in checks),
                             prefixes=['TEMPORARY'], postgresql_on_commit='DROP')
        temporary.create(connection)
        expected = {c['name']: c['sqltext'] for c in sa.inspect(connection).get_check_constraints(temporary.name)}
        actual = {c['name']: c['sqltext'] for c in inspector.get_check_constraints(table.name, schema='public')}
        incompatible |= expected != actual
        temporary.drop(connection)
    if incompatible:
        raise SchemaCompatibilityError(f'Incompatible schema for {table.name}; no tables were changed. '
                         'Apply a deliberate schema change before retrying.')


def initialize_schema(engine, metadata):
    """Create declared tables or atomically upgrade the recognized legacy schema."""
    with database_connection(engine, write=True) as connection:
        connection.exec_driver_sql('CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public')
        existing = set(sa.inspect(connection).get_table_names(schema='public'))
        missing = [table.name for table in metadata.sorted_tables if table.name not in existing]
        if 'documents' in existing:
            connection.exec_driver_sql('LOCK TABLE public.documents IN ACCESS EXCLUSIVE MODE')
            columns = {c['name']: c for c in sa.inspect(connection).get_columns('documents', schema='public')}
            if 'application_id' not in columns and 'applications' not in existing:
                from app.database.upgrade import upgrade_legacy
                from app.database.legacy_schema import build_metadata as legacy_metadata
                upgrade_legacy(connection, legacy_metadata(), columns)
            if 'embedding' in columns:
                from app.database.upgrade import split_documents
                split_documents(connection, metadata)
            elif 'name' not in columns:
                from app.database.upgrade import add_document_names
                add_document_names(connection, metadata)
        for table in metadata.sorted_tables:
            if table.name in sa.inspect(connection).get_table_names(schema='public'):
                validate_table(connection, table)
        metadata.create_all(connection)
    return {'tables_created': missing}
