"""Explicit transactional upgrades from the two supported chunk schemas."""
from dataclasses import asdict

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.schema import AddConstraint, CreateIndex

from app.models.application import Application


NEW_COLUMNS = ('application_id', 'embedding_model', 'embedding_size', 'provision_type')


def upgrade_legacy(connection, metadata, columns):
    from app.database.schema import validate_table

    dimensions = getattr(columns.get('embedding', {}).get('type'), 'dim', None)
    if type(dimensions) is not int or not 1 <= dimensions <= 16000:
        raise ValueError('Incompatible schema for documents; expected a fixed legacy vector size')
    documents = metadata.tables['public.documents']
    legacy = sa.Table('documents', sa.MetaData(schema='public'), *(
        sa.Column(c.name, Vector(dimensions) if c.name == 'embedding' else c.type,
                  nullable=c.nullable, primary_key=c.primary_key)
        for c in documents.c if c.name not in NEW_COLUMNS))
    validate_table(connection, legacy)
    applications = metadata.tables['public.applications']
    applications.create(connection)
    connection.exec_driver_sql('ALTER TABLE public.documents '
                               'ADD COLUMN application_id uuid, '
                               'ADD COLUMN embedding_model text, '
                               'ADD COLUMN embedding_size integer, '
                               'ADD COLUMN provision_type text')
    models = connection.execute(sa.select(documents.c.model_name).distinct()).scalars().all()
    for number, model in enumerate(sorted(models), 1):
        application = Application(name=f'Legacy application {number}', embedding_model=model,
                                  embedding_size=dimensions,
                                  data={'legacy': True, 'requires_model_configuration': not bool(model.strip())})
        connection.execute(applications.insert().values(**asdict(application)))
        connection.execute(documents.update().where(documents.c.model_name == model).values(
            application_id=application.id, embedding_model=model,
            embedding_size=dimensions, provision_type='local'))
    connection.exec_driver_sql('ALTER TABLE public.documents ALTER COLUMN embedding TYPE vector')
    for name in NEW_COLUMNS:
        # Identifiers come exclusively from the fixed declaration above.
        connection.exec_driver_sql(f'ALTER TABLE public.documents ALTER COLUMN {name} SET NOT NULL')
    for constraint in documents.constraints:
        if isinstance(constraint, (sa.ForeignKeyConstraint, sa.CheckConstraint)):
            connection.execute(AddConstraint(constraint, isolate_from_table=False))
    for index in documents.indexes:
        connection.execute(CreateIndex(index))


def split_documents(connection, metadata):
    """Replace the recognized chunk table atomically, retaining every chunk ID."""
    from uuid import uuid4
    from app.database.legacy_schema import build_metadata
    from app.database.schema import validate_table

    historical = build_metadata()
    for table in historical.sorted_tables:
        validate_table(connection, table)
    if sa.inspect(connection).has_table('embeddings', schema='public'):
        raise ValueError('Incompatible schema: embeddings already exists beside legacy documents')
    old = historical.tables['public.documents']
    rows = connection.execute(sa.select(old)).mappings().all()
    groups = {}
    for row in rows:
        key = (row['application_id'], row['source_id'] is None, row['source_id'] or row['id'],
               row['storage_key'], row['filename'], row['content_type'], row['source'])
        groups.setdefault(key, []).append(row)
    # No CASCADE: unknown external dependencies must abort rather than be destroyed.
    old.drop(connection)
    metadata.create_all(connection)
    documents = metadata.tables['public.documents']
    embeddings = metadata.tables['public.embeddings']
    for chunks in groups.values():
        first = chunks[0]
        document_id = uuid4()
        shared = {key: value for key, value in first['metadata'].items()
                  if key not in ('start', 'end', 'chunk_size', 'stride')
                  and all(row['metadata'].get(key) == value for row in chunks)}
        shared['source_id'] = str(first['source_id']) if first['source_id'] else None
        connection.execute(documents.insert().values(
            id=document_id, application_id=first['application_id'],
            storage_key=first['storage_key'], filename=first['filename'],
            content_type=first['content_type'], source=first['source'], metadata=shared,
            created_at=min(row['created_at'] for row in chunks),
            updated_at=max(row['updated_at'] for row in chunks)))
        connection.execute(embeddings.insert(), [
            {**{column.name: row[column.name] for column in embeddings.c if column.name != 'document_id'},
             'document_id': document_id} for row in chunks])
    if connection.scalar(sa.select(sa.func.count()).select_from(embeddings)) != len(rows):
        raise ValueError('Embedding migration count mismatch')
    if connection.scalar(sa.select(sa.func.count()).select_from(documents)) != len(groups):
        raise ValueError('Document migration count mismatch')
