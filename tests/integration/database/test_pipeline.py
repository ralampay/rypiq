import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from app.database.backends.postgresql import PostgreSQLBackend
from app.models.application import Application
from app.operations import document as document_operation
from app.storage.providers.local import LocalStorage


def test_ingestion_persists_model_name_and_compensates_failure(backend, tmp_path, monkeypatch):
    backend.create_database()
    application = backend.applications().create(Application(name="test", embedding_model="example.gguf", embedding_size=3))
    repository = backend.documents()
    storage = LocalStorage(tmp_path / 'objects')
    source = tmp_path / 'source.txt'
    source.write_text('example content')
    result = {'model': 'example.gguf', 'chunks': [
        {'text': 'example', 'embedding': [1., 0., 0.], 'index': 0, 'start': 0, 'end': 7},
    ]}
    monkeypatch.setattr(document_operation, 'embed_text_file', lambda **kwargs: result)
    documents = document_operation.ingest_document(str(source), storage, repository, 7, 7, application=application)
    stored = repository.get(documents[0].id)
    assert stored.model_name == 'example.gguf'
    assert (storage.root / stored.storage_key).read_text() == source.read_text()
    assert stored.metadata['start'] == 0
    before = [item.key for item in storage.list()]
    result['chunks'].append({'text': 'invalid', 'embedding': [1., 0.], 'index': 1, 'start': 7, 'end': 9})
    with pytest.raises(ValueError, match='Embedding'):
        document_operation.ingest_document(str(source), storage, repository, 7, 7, application=application)
    assert [item.id for item in repository.list()] == [stored.id]
    assert [item.key for item in storage.list()] == before


def test_real_model_cli_ingestion_search_and_reset(backend, tmp_path):
    project = Path(__file__).resolve().parents[3]
    model = Path(os.getenv('EMBEDDING_MODEL_PATH', project / 'models/embeddinggemma-300m-Q4_0.gguf')).resolve()
    if not model.is_file():
        pytest.skip('Embedding model not available for full CLI pipeline')
    settings = dict(backend.settings, dimensions=768)
    config = tmp_path / 'database.yml'
    config.write_text(yaml.safe_dump({'test': settings}))
    storage_config = tmp_path / 'storage.yml'
    storage_config.write_text(yaml.safe_dump({'test': {'adapter': 'local', 'root': str(tmp_path / 'objects')}}))
    environment = dict(os.environ, DATABASE_ENV='test', STORAGE_ENV='test',
                       RYPIQ_DATABASE_CONFIG=str(config), RYPIQ_STORAGE_CONFIG=str(storage_config))
    executable = str(Path(sys.executable).parent / 'rypiq')
    def invoke(*arguments):
        result = subprocess.run([executable, *arguments], cwd=tmp_path, env=environment,
                                capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    invoke('db:create')
    application = invoke('application:create', '--name', 'test', '--embedding-model', str(model))
    source = tmp_path / 'notes.txt'
    source.write_text('PostgreSQL stores structured data. Vector search retrieves related documents.')
    ingested = invoke('ingest', str(source), '--application-id', application['id'], '--chunk_size', '200', '--stride', '200', '--model', str(model))
    assert ingested['indexed_chunks'] == 1
    results = invoke('search', 'Which database stores structured data?', '--application-id', application['id'], '--model', str(model))
    assert results[0]['content'] == source.read_text()
    object_path = tmp_path / 'objects' / ingested['storage_key']
    with PostgreSQLBackend(settings) as initialized:
        assert initialized.documents().list()[0].model_name == model.name
    assert invoke('db:reset')['created'] is True
    assert object_path.read_text() == source.read_text()
    with PostgreSQLBackend(settings) as initialized:
        assert initialized.documents().list() == []
