import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from ragworkshop.storage.local import LocalStorage


def test_round_trip_metadata_collision_and_delete(tmp_path):
    source = tmp_path / "original.txt"
    source.write_bytes(b"python modules")
    storage = LocalStorage(tmp_path / "objects")
    obj = storage.save(source, "documents/lesson.txt", "text/plain")
    assert obj.checksum == hashlib.sha256(b"python modules").hexdigest()
    assert storage.list("documents/") == [obj]
    assert storage.exists(obj.key)
    destination = storage.get(obj.key, tmp_path / "retrieved.txt")
    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(FileExistsError):
        storage.save(source, obj.key, "text/plain")
    with pytest.raises(FileExistsError):
        storage.get(obj.key, destination)
    assert destination.read_bytes() == b"python modules"
    assert storage.delete(obj.key) is True
    assert storage.delete(obj.key) is False
    with pytest.raises(FileNotFoundError):
        storage.get(obj.key, tmp_path / "missing.txt")


@pytest.mark.parametrize("key", ["", "../escape", "/absolute", "a/../b", "a//b", "a/./b", "a\\b"])
def test_unsafe_keys(tmp_path, key):
    storage = LocalStorage(tmp_path / "objects")
    with pytest.raises(ValueError):
        storage.exists(key)


def test_symlink_escape(tmp_path):
    storage = LocalStorage(tmp_path / "objects")
    outside = tmp_path / "outside"
    outside.mkdir()
    bundle = storage.root / hashlib.sha256(b"key").hexdigest()
    bundle.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        storage.exists("key")


def test_save_failure_cleans_staging(monkeypatch, tmp_path):
    import ragworkshop.storage.local as local
    source = tmp_path / "source.txt"
    source.write_text("text")
    storage = LocalStorage(tmp_path / "objects")
    def fail(*args):
        raise OSError("disk failure")
    monkeypatch.setattr(local.os, "rename", fail)
    with pytest.raises(OSError):
        storage.save(source, "key", "text/plain")
    assert list(storage.root.iterdir()) == []
    with pytest.raises(FileNotFoundError):
        storage.save(tmp_path / "missing", "key", "text/plain")


def test_storage_cli_then_document_processing(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("python modules", encoding="utf-8")
    def run(*args):
        result = subprocess.run(["rag-workshop", *map(str, args)], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    saved = run("storage:save", source, "documents/source.txt", "--content-type", "text/plain")
    assert run("storage:list")[0] == saved
    destination = tmp_path / "downloaded.txt"
    run("storage:get", saved["key"], destination)
    assert run("process-file", destination)["chunks"][0]["text"] == "python modules"
    assert run("storage:exists", saved["key"]) == {"exists": True}
    assert run("storage:delete", saved["key"]) == {"deleted": True}
