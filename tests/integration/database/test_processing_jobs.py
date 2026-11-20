from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import Mock
from uuid import uuid4

import pytest
import sqlalchemy as sa

from app.exceptions import ConflictError, DocumentNotFoundError, ValidationError
from app.models.application import Application
from app.operations.documents.create import CreateDocument
from app.operations.documents.process import ProcessDocument
from app.operations.processing_jobs.run import RunProcessingJob
from app.storage.providers.local import LocalStorage


@pytest.fixture
def queued(backend, tmp_path):
    backend.create_database()
    value = backend.applications().create(Application(name='queue', embedding_model='fake', embedding_size=2,
        data={'chunking': {'chunk_size': 5, 'stride': 5}}))
    storage = LocalStorage(tmp_path / 'objects')
    source = tmp_path / 'report.txt'
    source.write_text('abcdefghij')
    document = CreateDocument(name='Report', path=str(source), application=value,
                             repository=backend.documents(), storage=storage).execute().document
    return value, document, storage


def expire(backend, job):
    table = backend.metadata.tables['public.document_processing_jobs']
    with backend.engine.begin() as connection:
        connection.execute(table.update().where(table.c.id == job.id).values(
            lease_expires_at=sa.func.clock_timestamp() - sa.text("INTERVAL '1 second'")))


def process(backend, storage, job, provider=None):
    if provider is None:
        provider = Mock(model_name='fake')
        provider.embed.side_effect = lambda texts: [[1., 0.] for _ in texts]
    return ProcessDocument(document_id=job.document_id, batch_size=job.batch_size,
        repository=backend.processing_documents(job), applications=backend.applications(), storage=storage,
        provider_scope=lambda _: nullcontext(provider)).execute()


def test_enqueue_deduplicates_concurrent_requests_and_restricts_delete(backend, queued):
    _, document, _ = queued
    jobs = backend.processing_jobs()
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: jobs.enqueue(document_id=document.id), range(8)))
    assert len({job.id for job in results}) == 1
    assert jobs.latest(document.id).state == 'queued'
    with pytest.raises(ConflictError, match='active'):
        backend.documents().delete(document.id)
    with pytest.raises(ConflictError):
        with backend.documents().processing_lock(document.id):
            pytest.fail('CLI must not steal queued work')
    with pytest.raises(DocumentNotFoundError):
        jobs.enqueue(document_id=uuid4())
    with pytest.raises(ValidationError):
        jobs.enqueue(document_id=document.id, batch_size=0)


def test_competing_claims_and_atomic_success(backend, queued):
    _, document, storage = queued
    jobs = backend.processing_jobs()
    jobs.enqueue(document_id=document.id)
    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(lambda _: jobs.claim(), range(2)))
    assert sum(job is not None for job in claims) == 1
    job = next(job for job in claims if job)
    assert jobs.heartbeat(job)
    result = process(backend, storage, job)
    assert len(result.embeddings) == 2
    assert result.document.status == 'done'
    assert jobs.get(job.id).state == 'succeeded'
    assert not jobs.fail(job, message='late cleanup error')
    assert backend.documents().get(document.id).status == 'done'
    assert backend.documents().delete(document.id)
    assert jobs.get(job.id) is None
    assert storage.exists(document.storage_key)


def test_expired_claim_cannot_write_or_renew_and_recovers(backend, queued):
    _, document, storage = queued
    jobs = backend.processing_jobs()
    jobs.enqueue(document_id=document.id)
    first = jobs.claim()
    backend.processing_documents(first).set_status(document.id, 'processing')
    expire(backend, first)
    assert not jobs.heartbeat(first)
    second = jobs.claim()
    assert second.id == first.id and second.claim_token != first.claim_token
    assert second.attempts == 2
    with pytest.raises(ConflictError, match='expired'):
        backend.processing_documents(first).set_status(document.id, 'error')
    assert not jobs.fail(first, message='stale failure')
    assert process(backend, storage, second).document.status == 'done'


def test_lost_claim_during_embedding_preserves_previous_chunks(backend, queued):
    _, document, storage = queued
    jobs = backend.processing_jobs()
    jobs.enqueue(document_id=document.id)
    original = process(backend, storage, jobs.claim())
    jobs.enqueue(document_id=document.id)
    stale = jobs.claim()
    provider = Mock(model_name='fake')
    def embed(texts):
        expire(backend, stale)
        return [[0., 1.] for _ in texts]
    provider.embed.side_effect = embed
    with pytest.raises(ConflictError, match='expired'):
        process(backend, storage, stale, provider)
    assert backend.documents().details(document.id).embeddings == original.embeddings
    assert process(backend, storage, jobs.claim()).document.status == 'done'


def test_three_interruptions_exhaust_job_and_explicit_retry_is_new(backend, queued):
    _, document, _ = queued
    jobs = backend.processing_jobs()
    original = jobs.enqueue(document_id=document.id)
    for attempt in range(1, 4):
        claimed = jobs.claim()
        assert claimed.attempts == attempt
        expire(backend, claimed)
    assert jobs.claim() is None
    assert jobs.get(original.id).state == 'failed'
    assert backend.documents().get(document.id).status == 'error'
    assert jobs.enqueue(document_id=document.id).id != original.id


def test_processing_failure_is_terminal_and_sanitized(backend, queued):
    _, document, storage = queued
    jobs = backend.processing_jobs()
    job = jobs.enqueue(document_id=document.id)
    provider = Mock(model_name='fake')
    provider.embed.side_effect = RuntimeError('SECRET UPSTREAM URL')
    command = RunProcessingJob(repository=jobs, process=lambda job: process(backend, storage, job, provider),
                               heartbeat=lambda _: nullcontext())
    assert command.execute()
    assert jobs.get(job.id).state == 'failed'
    assert 'SECRET' not in jobs.get(job.id).error
    assert backend.documents().get(document.id).status == 'error'
    assert storage.exists(document.storage_key)
    assert not command.execute()


def test_completion_failure_rolls_back_embeddings_and_document_state(backend, queued, monkeypatch):
    _, document, storage = queued
    jobs = backend.processing_jobs()
    jobs.enqueue(document_id=document.id)
    original = process(backend, storage, jobs.claim())
    jobs.enqueue(document_id=document.id)
    claimed = jobs.claim()
    from app.database.repositories.postgresql import processing_job
    def fail(*_):
        raise RuntimeError('completion transaction failed')
    monkeypatch.setattr(processing_job, 'complete_claim', fail)
    with pytest.raises(RuntimeError):
        process(backend, storage, claimed)
    assert jobs.get(claimed.id).state == 'running'
    assert backend.documents().details(document.id).embeddings == original.embeddings
    assert backend.documents().get(document.id).status == 'error'


def test_worker_waits_for_existing_document_lock(backend, queued):
    _, document, storage = queued
    jobs = backend.processing_jobs()
    with backend.documents().processing_lock(document.id):
        job = jobs.enqueue(document_id=document.id)
        command = RunProcessingJob(repository=jobs, process=lambda job: process(backend, storage, job),
                                   heartbeat=lambda _: nullcontext())
        assert command.execute()
        assert jobs.get(job.id).state == 'queued'
        assert jobs.get(job.id).attempts == 0
    assert command.execute()
    assert jobs.get(job.id).state == 'succeeded'


@pytest.mark.parametrize('change', [
    'DROP INDEX uq_processing_jobs_active_document',
    "DROP INDEX uq_processing_jobs_active_document; CREATE UNIQUE INDEX uq_processing_jobs_active_document ON document_processing_jobs (document_id) WHERE state = 'running'",
])
def test_migration_rejects_missing_or_incorrect_queue_uniqueness(backend, change):
    from app.exceptions import SchemaCompatibilityError
    backend.create_database()
    with backend.engine.begin() as connection:
        connection.exec_driver_sql(change)
    with pytest.raises(SchemaCompatibilityError):
        backend.migrate()


def test_busy_document_does_not_starve_other_jobs(backend, queued, tmp_path):
    value, document, storage = queued
    source = tmp_path / 'second.txt'
    source.write_text('second')
    second = CreateDocument(name='Second', path=str(source), application=value,
                            repository=backend.documents(), storage=storage).execute().document
    jobs = backend.processing_jobs()
    with backend.documents().processing_lock(document.id):
        jobs.enqueue(document_id=document.id)
        other = jobs.enqueue(document_id=second.id)
        command = RunProcessingJob(repository=jobs, process=lambda job: process(backend, storage, job),
                                   heartbeat=lambda _: nullcontext())
        command.execute()
        assert jobs.claim().id == other.id


def test_enqueue_and_delete_race_has_no_orphan_accepted_job(backend, queued):
    from threading import Barrier
    _, document, _ = queued
    barrier = Barrier(2)
    jobs = backend.processing_jobs()
    def enqueue():
        barrier.wait(timeout=5)
        try:
            return jobs.enqueue(document_id=document.id)
        except DocumentNotFoundError:
            return None
    def delete():
        barrier.wait(timeout=5)
        try:
            return backend.documents().delete(document.id)
        except ConflictError:
            return False
    with ThreadPoolExecutor(max_workers=2) as executor:
        enqueued, deleted = executor.submit(enqueue), executor.submit(delete)
        job, removed = enqueued.result(timeout=10), deleted.result(timeout=10)
    if removed:
        assert job is None
        assert jobs.latest(document.id) is None
    else:
        assert job is not None
        assert backend.documents().get(document.id) is not None
        assert jobs.get(job.id).state == 'queued'
