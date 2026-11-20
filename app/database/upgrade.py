"""Explicit transactional upgrades from the two supported chunk schemas."""
from dataclasses import asdict

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.schema import AddConstraint, CreateIndex

from app.exceptions import SchemaCompatibilityError
from app.models.application import Application


NEW_COLUMNS = ('application_id', 'embedding_model', 'embedding_size', 'provision_type')


def upgrade_legacy(connection, metadata, columns):
    from app.database.schema import validate_table

    dimensions = getattr(columns.get('embedding', {}).get('type'), 'dim', None)
    if type(dimensions) is not int or not 1 <= dimensions <= 16000:
        raise SchemaCompatibilityError('Incompatible schema for documents; expected a fixed legacy vector size')
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
        values = asdict(application)
        values['provision_type'] = values.pop('embedding_provision_type')
        values.pop('inference_provision_type')
        connection.execute(applications.insert().values(**values))
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
        raise SchemaCompatibilityError('Incompatible schema: embeddings already exists beside legacy documents')
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
    parents = []
    children = []
    for chunks in groups.values():
        first = chunks[0]
        document_id = uuid4()
        shared = {key: value for key, value in first['metadata'].items()
                  if key not in ('start', 'end', 'chunk_size', 'stride')
                  and all(row['metadata'].get(key) == value for row in chunks)}
        shared['source_id'] = str(first['source_id']) if first['source_id'] else None
        parents.append(dict(
            id=document_id, application_id=first['application_id'], status='done',
            storage_key=first['storage_key'], filename=first['filename'],
            content_type=first['content_type'], source=first['source'], metadata=shared,
            created_at=min(row['created_at'] for row in chunks),
            updated_at=max(row['updated_at'] for row in chunks)))
        children.extend([
            {**{column.name: row[column.name] for column in embeddings.c if column.name != 'document_id'},
             'document_id': document_id} for row in chunks])
    from app.models.processing import SNAPSHOT_KEY, processing_settings
    applications = {row['id']: _historical_application(row) for row in connection.execute(
        sa.select(metadata.tables['public.applications'])).mappings()}
    for parent in parents:
        parent['metadata'][SNAPSHOT_KEY] = processing_settings(applications[parent['application_id']])
    assign_document_names(parents)
    if parents:
        connection.execute(documents.insert(), parents)
        connection.execute(embeddings.insert(), children)
    if connection.scalar(sa.select(sa.func.count()).select_from(embeddings)) != len(rows):
        raise SchemaCompatibilityError('Embedding migration count mismatch')
    if connection.scalar(sa.select(sa.func.count()).select_from(documents)) != len(groups):
        raise SchemaCompatibilityError('Document migration count mismatch')


def assign_document_names(rows: list[dict]) -> None:
    """Reserve unique filenames first, then allocate collision-safe names per application."""
    from collections import Counter, defaultdict

    applications = defaultdict(list)
    for row in rows:
        applications[row['application_id']].append(row)
    for documents in applications.values():
        counts = Counter(row['filename'].strip() for row in documents)
        reserved = {name for name, count in counts.items() if name and count == 1}
        for row in sorted(documents, key=lambda row: row['id']):
            base = row['filename'].strip()
            if base and counts[base] == 1:
                row['name'] = base
                continue
            candidate = f"{base} ({row['id']})" if base else f"Document {row['id']}"
            name, number = candidate, 2
            while name in reserved:
                name = f'{candidate} ({number})'
                number += 1
            row['name'] = name
            reserved.add(name)


def add_document_names(connection, metadata):
    """Upgrade only the recognized unnamed parent schema, preserving IDs and timestamps."""
    from app.database.unnamed_schema import build_metadata
    from app.database.schema import validate_table

    historical = build_metadata()
    for table in historical.sorted_tables:
        if sa.inspect(connection).has_table(table.name, schema='public'):
            validate_table(connection, table)
    old = historical.tables['public.documents']
    rows = [dict(row) for row in connection.execute(sa.select(
        old.c.id, old.c.application_id, old.c.filename)).mappings()]
    assign_document_names(rows)
    connection.exec_driver_sql('ALTER TABLE public.documents ADD COLUMN name text')
    documents = metadata.tables['public.documents']
    for row in rows:
        connection.execute(documents.update().where(documents.c.id == row['id']).values(name=row['name']))
    connection.exec_driver_sql('ALTER TABLE public.documents ALTER COLUMN name SET NOT NULL')
    for constraint in documents.constraints:
        if constraint.name in ('uq_documents_application_name', 'ck_documents_name_nonempty'):
            connection.execute(AddConstraint(constraint, isolate_from_table=False))


def add_document_status(connection, metadata):
    """Upgrade the recognized parent schema, including a search settings snapshot."""
    from app.database.pre_status_schema import build_metadata
    from app.database.schema import validate_table
    from app.models.processing import SNAPSHOT_KEY, processing_settings

    if 'status' in {c['name'] for c in sa.inspect(connection).get_columns('documents', schema='public')}:
        return
    historical = build_metadata()
    for table in historical.sorted_tables:
        if sa.inspect(connection).has_table(table.name, schema='public'):
            validate_table(connection, table)
    connection.exec_driver_sql("ALTER TABLE public.documents ADD COLUMN status text NOT NULL DEFAULT 'pending'")
    documents = metadata.tables['public.documents']
    embeddings = metadata.tables['public.embeddings']
    has_embeddings = sa.inspect(connection).has_table('embeddings', schema='public')
    applications = metadata.tables['public.applications']
    for row in connection.execute(sa.select(applications)).mappings():
        snapshot = processing_settings(_historical_application(row))
        owned = documents.c.application_id == row['id']
        if has_embeddings:
            connection.execute(documents.update().where(owned, sa.exists().where(
                embeddings.c.document_id == documents.c.id)).values(status='done'))
        for document in connection.execute(sa.select(documents).where(owned)).mappings():
            connection.execute(documents.update().where(documents.c.id == document['id']).values(
                metadata={**document['metadata'], SNAPSHOT_KEY: snapshot}))
    for constraint in documents.constraints:
        if constraint.name == 'ck_documents_status':
            connection.execute(AddConstraint(constraint, isolate_from_table=False))
    for index in documents.indexes:
        if index.name == 'ix_documents_status':
            connection.execute(CreateIndex(index))


def _historical_application(row):
    """Convert frozen schema rows without adding a legacy public model field."""
    values = dict(row)
    provider = values.pop('provision_type')
    return Application(**values, embedding_provision_type=provider, inference_provision_type=provider)


def split_application_providers(connection, metadata):
    """Atomically replace the recognized shared application provider field."""
    from app.database.schema import validate_table
    from app.database.shared_provider_schema import build_metadata

    connection.exec_driver_sql('LOCK TABLE public.applications IN ACCESS EXCLUSIVE MODE')
    columns = {c['name'] for c in sa.inspect(connection).get_columns('applications', schema='public')}
    if 'provision_type' not in columns:
        return  # Final schema validation still rejects partial/unknown layouts.
    for table in build_metadata().sorted_tables:
        if sa.inspect(connection).has_table(table.name, schema='public'):
            validate_table(connection, table)
    connection.exec_driver_sql('ALTER TABLE public.applications '
                              'ADD COLUMN embedding_provision_type text, '
                              'ADD COLUMN inference_provision_type text')
    connection.exec_driver_sql('UPDATE public.applications SET '
                              'embedding_provision_type = provision_type, '
                              'inference_provision_type = provision_type')
    connection.exec_driver_sql('ALTER TABLE public.applications '
                              'ALTER COLUMN embedding_provision_type SET NOT NULL, '
                              'ALTER COLUMN inference_provision_type SET NOT NULL, '
                              'DROP CONSTRAINT ck_applications_provider, '
                              'DROP COLUMN provision_type')
    for constraint in metadata.tables['public.applications'].constraints:
        if constraint.name in ('ck_applications_embedding_provider', 'ck_applications_inference_provider'):
            connection.execute(AddConstraint(constraint, isolate_from_table=False))
