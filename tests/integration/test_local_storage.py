import pytest

from app.storage.providers.local import LocalStorage
from app.operations.storage import save_file


@pytest.fixture
def storage(tmp_path):
    return LocalStorage(tmp_path / "objects")


@pytest.mark.integration
def test_save_and_list(storage, tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("Hello RAG", encoding="utf-8")

    obj = save_file(source, storage)

    assert obj.size == len(b"Hello RAG")
    assert obj.key.startswith("documents/")

    objects = storage.list()

    assert len(objects) == 1
    assert objects[0].key == obj.key


@pytest.mark.integration
def test_get_file(storage, tmp_path):
    source = tmp_path / "original.txt"
    source.write_text("Stored content")

    obj = save_file(source, storage)

    target = tmp_path / "downloaded.txt"
    storage.get(obj.key, target)

    assert target.read_text() == "Stored content"


@pytest.mark.integration
def test_delete_file(storage, tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("Delete me")

    obj = save_file(source, storage)

    assert storage.exists(obj.key)
    assert storage.delete(obj.key)
    assert not storage.exists(obj.key)


@pytest.mark.integration
def test_reject_path_traversal(storage, tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("Test")

    with pytest.raises(ValueError):
        storage.save(
            source,
            "../outside.txt",
            "text/plain",
        )


@pytest.mark.integration
def test_duplicate_key(storage, tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("First")

    storage.save(source, "test/file.txt", "text/plain")

    with pytest.raises(FileExistsError):
        storage.save(
            source,
            "test/file.txt",
            "text/plain",
        )
