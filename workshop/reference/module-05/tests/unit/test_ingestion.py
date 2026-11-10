from pathlib import Path
import pytest

from ragworkshop.context import ApplicationContext
from ragworkshop.embeddings.fake import FakeEmbeddingProvider
from ragworkshop.operations import ingestion
from ragworkshop.operations.search import semantic_search
from ragworkshop.processors.chunks import TextChunker
from ragworkshop.storage.base import StoredObject


class MemoryStorage:
    def __init__(self):
        self.keys = set()
    def save(self, source, key, content_type):
        self.keys.add(key)
        return StoredObject(key, 1, content_type, "checksum")
    def delete(self, key):
        self.keys.remove(key)
        return True


class FailingRepository:
    def __init__(self):
        self.records = []
        self.deleted = []
    def ensure_space(self, model_name, dimensions):
        pass
    def add(self, documents):
        self.records = documents[:1]
        raise RuntimeError("partial database write")
    def delete(self, ids):
        self.deleted = ids
        self.records = [doc for doc in self.records if doc.id not in ids]


def test_partial_failure_removes_all_allocated_ids(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("python cat")
    storage = MemoryStorage()
    repo = FailingRepository()
    context = ApplicationContext(repo, storage, FakeEmbeddingProvider())
    with pytest.raises(RuntimeError, match="partial"):
        ingestion.ingest_document(path, context, TextChunker(5, 5))
    assert len(repo.deleted) == 2
    assert repo.records == []
    assert storage.keys == set()


def test_cleanup_failure_keeps_original_and_attempts_storage(tmp_path):
    path = tmp_path / "input.txt"
    path.write_text("python")
    storage = MemoryStorage()
    repo = FailingRepository()
    def fail(ids):
        raise OSError("cleanup unavailable")
    repo.delete = fail
    with pytest.raises(ExceptionGroup) as error:
        ingestion.ingest_document(path, ApplicationContext(repo, storage, FakeEmbeddingProvider()), TextChunker())
    assert isinstance(error.value.exceptions[0], RuntimeError)
    assert isinstance(error.value.exceptions[1], OSError)
    assert storage.keys == set()


def test_empty_file_compensates(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text(" ")
    storage = MemoryStorage()
    with pytest.raises(ValueError, match="no extractable"):
        ingestion.ingest_document(path, ApplicationContext(FailingRepository(), storage, FakeEmbeddingProvider()), TextChunker())
    assert storage.keys == set()


@pytest.mark.parametrize("query,limit", [(" ", 1), ("cat", 0), ("cat", True)])
def test_invalid_search(query, limit):
    with pytest.raises(ValueError):
        semantic_search(query, ApplicationContext(FailingRepository(), MemoryStorage(), FakeEmbeddingProvider()), limit)
