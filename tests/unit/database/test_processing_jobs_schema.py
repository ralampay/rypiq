from app.database.schema import build_metadata


def test_durable_queue_has_single_active_job_constraint():
    metadata = build_metadata()
    assert 'public.document_processing_jobs' in metadata.tables
    table = metadata.tables['public.document_processing_jobs']
    assert {'claim_token', 'lease_expires_at', 'attempts', 'batch_size', 'error'} <= set(table.c.keys())
    assert any(index.unique and index.dialect_options['postgresql']['where'] is not None
               for index in table.indexes)
