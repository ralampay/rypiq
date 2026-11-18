"""Contracts for parent documents and independently searchable chunks."""
from unittest.mock import Mock

import pytest

from app.database.schema import build_metadata
from app.models.application import Application
from app.operations.documents.persist import persist_document


def test_schema_has_parent_and_chunk_tables():
    metadata = build_metadata()
    assert 'public.embeddings' in metadata.tables
    documents = metadata.tables['public.documents']
    embeddings = metadata.tables['public.embeddings']
    assert 'embedding' not in documents.c
    assert 'content' not in documents.c
    assert not embeddings.c.document_id.nullable
    assert next(iter(embeddings.c.document_id.foreign_keys)).ondelete == 'CASCADE'
    assert str(embeddings.c.metadata.type) == 'JSONB'


def test_ingestion_writes_one_aggregate_and_returns_details(tmp_path):
    source = tmp_path / 'source.txt'
    source.write_text('abcd')
    storage = Mock()
    storage.save.return_value = Mock(key='owned', content_type='text/plain', checksum='checksum')
    repository = Mock()
    result = persist_document(str(source), storage, repository, 2, 2,
                              {'model': 'model', 'chunks': [
                                  {'text': 'ab', 'embedding': [1., 0.], 'index': 0, 'start': 0, 'end': 2},
                                  {'text': 'cd', 'embedding': [0., 1.], 'index': 1, 'start': 2, 'end': 4},
                              ]}, Application(name='test', embedding_model='model', embedding_size=2))
    repository.create_with_embeddings.assert_called_once()
    assert [item.content for item in result.embeddings] == ['ab', 'cd']
    assert all(item.document_id == result.document.id for item in result.embeddings)
    assert result.document.metadata['storage_checksum'] == 'checksum'
    storage.delete.assert_not_called()


def test_aggregate_rejects_foreign_chunk_before_connection():
    from uuid import uuid4
    from app.models.document import Document
    from app.models.embedding import Embedding
    from app.database.repositories.postgresql import PostgreSQLDocumentRepository

    engine = Mock()
    repository = PostgreSQLDocumentRepository(engine, build_metadata().tables['public.documents'])
    parent = Document(application_id=uuid4())
    chunk = Embedding(document_id=uuid4(), content='text', embedding=[1., 0.], embedding_size=2)
    with pytest.raises(ValueError, match='another document'):
        repository.create_with_embeddings(parent, [chunk])
    engine.begin.assert_not_called()
