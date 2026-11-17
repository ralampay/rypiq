from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from threading import Barrier, Event

import pytest

from app.storage.providers import local
from app.storage.providers.local import LocalStorage
from app.operations.storage import save_file


@pytest.mark.parametrize('key', ['', 'a\\b', '.', '/absolute', '../escape'])
def test_invalid_keys(tmp_path, key):
    storage = LocalStorage(tmp_path / 'objects')
    with pytest.raises(ValueError):
        storage.exists(key)


def test_missing_paths_and_existing_download_are_safe(tmp_path):
    storage = LocalStorage(tmp_path / 'objects')
    assert storage.list() == []
    assert storage.delete('missing') is False
    with pytest.raises(FileNotFoundError):
        storage.get('missing', tmp_path / 'download')
    with pytest.raises(FileNotFoundError):
        storage.save(tmp_path / 'missing', 'key', 'text/plain')
    with pytest.raises(FileNotFoundError):
        save_file(tmp_path / 'missing', storage)
    source = tmp_path / 'source'
    source.write_text('data')
    storage.save(source, 'a/file', 'text/plain')
    with pytest.raises(FileExistsError):
        storage.get('a/file', source)
    assert source.read_text() == 'data'
    assert storage.list('other') == []
    assert [item.key for item in storage.list('a')] == ['a/file']


def test_symlink_escape_is_rejected_and_links_are_not_listed(tmp_path):
    storage = LocalStorage(tmp_path / 'objects')
    storage.root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (storage.root / 'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError):
        storage.exists('escape/file')
    assert storage.list() == []


def test_publication_failure_cleans_staging_and_allows_retry(tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path / 'objects')
    source = tmp_path / 'source'
    source.write_bytes(b'complete data')
    def fail(staged, destination):
        assert staged.read_bytes() == source.read_bytes()
        assert not destination.exists()
        raise OSError('publication failed')
    with monkeypatch.context() as patch:
        patch.setattr(local.os, 'link', fail)
        with pytest.raises(OSError, match='publication failed'):
            storage.save(source, 'file', 'application/octet-stream')
    assert not storage.exists('file')
    assert list(storage.root.iterdir()) == []
    storage.save(source, 'file', 'application/octet-stream')
    assert (storage.root / 'file').read_bytes() == source.read_bytes()


def test_reader_only_sees_complete_published_content(tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path / 'objects')
    source = tmp_path / 'source'
    payload = b'data' * 500000
    source.write_bytes(payload)
    ready, release = Event(), Event()
    real_link = local.os.link
    def pause(staged, destination):
        ready.set()
        assert release.wait(5)
        return real_link(staged, destination)
    monkeypatch.setattr(local.os, 'link', pause)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(storage.save, source, 'file', 'application/octet-stream')
        try:
            assert ready.wait(5)
            assert not storage.exists('file')
        finally:
            release.set()
        result = future.result(timeout=5)
    assert (storage.root / 'file').read_bytes() == payload
    assert result.size == len(payload)
    assert result.checksum == hashlib.sha256(payload).hexdigest()


def test_competing_writers_do_not_overwrite_or_remove_winner(tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path / 'objects')
    sources = [tmp_path / 'one', tmp_path / 'two']
    sources[0].write_bytes(b'A')
    sources[1].write_bytes(b'BB')
    barrier = Barrier(2)
    real_link = local.os.link
    def synchronized_link(staged, destination):
        barrier.wait(5)
        return real_link(staged, destination)
    monkeypatch.setattr(local.os, 'link', synchronized_link)
    def write(source):
        try:
            return storage.save(source, 'file', 'text/plain')
        except FileExistsError as exc:
            return exc
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(write, sources))
    winners = [(i, result) for i, result in enumerate(outcomes) if not isinstance(result, FileExistsError)]
    assert len(winners) == 1
    index, result = winners[0]
    assert (storage.root / 'file').read_bytes() == sources[index].read_bytes()
    assert result.checksum == hashlib.sha256(sources[index].read_bytes()).hexdigest()
    assert list(storage.root.iterdir()) == [storage.root / 'file']


def test_staging_read_failure_leaves_no_object(tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path / 'objects')
    source = tmp_path / 'source'
    source.write_bytes(b'content')
    original = Path.open
    def fail(path, *args, **kwargs):
        if path == source and args == ('rb',):
            raise OSError('source read failed')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', fail)
    with pytest.raises(OSError, match='source read failed'):
        storage.save(source, 'file', 'text/plain')
    assert not storage.exists('file')
    assert list(storage.root.iterdir()) == []


@pytest.mark.parametrize('publication_fails', [False, True])
def test_cleanup_failure_preserves_commit_state(tmp_path, monkeypatch, publication_fails):
    storage = LocalStorage(tmp_path / 'objects')
    source = tmp_path / 'source'
    source.write_bytes(b'content')
    real_unlink = Path.unlink
    primary = OSError('publication failed')
    def fail_cleanup(path, *args, **kwargs):
        if path.name.startswith('.rypiq-'):
            raise OSError('cleanup failed')
        return real_unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'unlink', fail_cleanup)
        if publication_fails:
            def fail_link(*args):
                raise primary
            patch.setattr(local.os, 'link', fail_link)
            with pytest.raises(OSError) as error:
                storage.save(source, 'file', 'text/plain')
            assert error.value is primary
            assert primary.__notes__
            assert not storage.exists('file')
        else:
            with pytest.raises(RuntimeError, match='published'):
                storage.save(source, 'file', 'text/plain')
            assert (storage.root / 'file').read_bytes() == b'content'
    for path in storage.root.glob('.rypiq-*'):
        path.unlink()


def test_staging_creation_failure_has_no_published_object(tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path / 'objects')
    source = tmp_path / 'source'
    source.write_bytes(b'data')
    def fail(**kwargs):
        raise OSError('staging unavailable')
    monkeypatch.setattr(local.tempfile, 'NamedTemporaryFile', fail)
    with pytest.raises(OSError, match='staging unavailable'):
        storage.save(source, 'file', 'text/plain')
    assert not storage.exists('file')
