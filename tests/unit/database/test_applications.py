from dataclasses import fields

import pytest

from app.database.schema import build_metadata
from app.models.document import Document
from app.models.embedding import Embedding


def test_application_owned_dynamic_schema():
    metadata = build_metadata()
    assert 'public.applications' in metadata.tables
    documents = metadata.tables['public.documents']
    assert metadata.tables['public.embeddings'].c.embedding.type.dim is None
    assert not documents.c.application_id.nullable
    assert documents.c.application_id.foreign_keys
    assert {'embedding_model', 'embedding_size', 'provision_type'} <= {
        field.name for field in fields(Embedding)
    }


def test_application_defaults_and_key_redaction():
    from app.models.application import Application
    from app.cli_handlers import application_json

    first = Application(name='one', embedding_model='model')
    second = Application(name='two', embedding_model='model')
    first.data['nested'] = {'value': 1}
    assert second.data == {}
    assert first.api_key != second.api_key
    assert len(first.api_key) >= 43
    assert first.api_key not in repr(first)
    assert 'api_key' not in application_json(first)
    assert application_json(first, include_key=True)['api_key'] == first.api_key


@pytest.mark.parametrize('size', [True, 0, 16001, 1.0])
def test_document_size_rejected_before_database_io(size):
    from unittest.mock import Mock
    from uuid import uuid4
    from app.database.repositories.postgresql import PostgreSQLEmbeddingRepository

    engine = Mock()
    repository = PostgreSQLEmbeddingRepository(engine, build_metadata().tables['public.embeddings'])
    with pytest.raises(ValueError, match='Embedding size'):
        repository.create(Embedding(content='test', embedding=[1.], embedding_size=size, document_id=uuid4()))
    engine.begin.assert_not_called()
