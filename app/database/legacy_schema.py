"""Frozen application-owned chunk schema, used only for deliberate upgrades."""
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, UUID


def build_metadata(dimensions=768):
    if type(dimensions) is not int or not 1 <= dimensions <= 16000:
        raise ValueError('dimensions must be an integer between 1 and 16000')
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
        sa.Column('id', UUID(as_uuid=True), primary_key=True),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('embedding', Vector(), nullable=False),
        sa.Column('application_id', UUID(as_uuid=True),
                  sa.ForeignKey('public.applications.id', ondelete='RESTRICT'), nullable=False, index=True),
        sa.Column('embedding_model', sa.Text(), nullable=False),
        sa.Column('embedding_size', sa.Integer(), nullable=False),
        sa.Column('provision_type', sa.Text(), nullable=False),
        sa.CheckConstraint('vector_dims(embedding) = embedding_size', name='ck_documents_vector_size'),
        sa.CheckConstraint('embedding_size BETWEEN 1 AND 16000', name='ck_documents_size'),
        sa.CheckConstraint("provision_type IN ('local', 'openai', 'llama-server')",
                           name='ck_documents_provider'),
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

