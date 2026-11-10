import pytest
from ragworkshop.database.factory import get_database
from ragworkshop.database.models import Document
from ragworkshop.database.migrations import migrate


def test_persistence_filters_and_migrations(tmp_path):
    path = tmp_path / "persistent"
    backend = get_database("chroma", path)
    assert migrate(backend) == ["0001"]
    assert migrate(backend) == []
    repository = backend.documents()
    repository.ensure_space("test-space", 2)
    assert repository.search_by_vector([1.0, 0.0]) == []
    repository.add([Document("id-1", "source-1", "python", [1.0, 0.0], "documents/a.txt", {"page": 1}),
                    Document("id-2", "source-2", "cat", [0.0, 1.0], "documents/b.txt", {"page": 2})])
    fresh = get_database("chroma", path).documents()
    fresh.ensure_space("test-space", 2)
    result = fresh.search_by_vector([1.0, 0.0], filters={"source_id": "source-1", "page": 1})
    assert [match.document.id for match in result] == ["id-1"]
    assert result[0].distance == pytest.approx(0.0)
    assert result[0].document.metadata == {"page": 1}
    with pytest.raises(ValueError, match="space mismatch"):
        fresh.ensure_space("another-model", 2)
    with pytest.raises(ValueError, match="space mismatch"):
        fresh.ensure_space("test-space", 3)
    with pytest.raises(ValueError, match="dimension"):
        fresh.search_by_vector([1.0])
    fresh.delete(["id-1", "id-2"])
    assert fresh.search_by_vector([1.0, 0.0]) == []


def test_backend_factory_rejects_unknown(tmp_path):
    with pytest.raises(ValueError):
        get_database("postgres", tmp_path)
