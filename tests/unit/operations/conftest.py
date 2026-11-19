from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from shutil import copyfile
from unittest.mock import Mock

import pytest

from app.models.document import DocumentDetails


@pytest.fixture
def document_resources():
    """Configure lifecycle-capable repository/storage mocks for operation tests."""
    def configure(repository=None, storage=None, source=None):
        repository = repository if repository is not None else Mock()
        storage = storage if storage is not None else Mock()
        records = {}
        def create(document):
            records[document.id] = replace(document)
            return replace(document)
        def status(identifier, value):
            records[identifier] = replace(records[identifier], status=value)
            return replace(records[identifier])
        def complete(document, chunks, *, application):
            return DocumentDetails(status(document.id, 'done'), chunks)
        repository.name_exists.return_value = False
        repository.create.side_effect = create
        repository.update.side_effect = create
        repository.set_status.side_effect = status
        repository.replace_embeddings.side_effect = complete
        repository.processing_lock.side_effect = lambda identifier: nullcontext()
        storage.save.return_value = Mock(key='stored.txt', content_type='text/plain', checksum='checksum')
        if source is not None:
            storage.get.side_effect = lambda key, destination: Path(copyfile(source, destination))
        return repository, storage
    return configure
