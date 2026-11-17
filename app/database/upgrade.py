"""One explicit, transactional upgrade from the original documents schema."""
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
