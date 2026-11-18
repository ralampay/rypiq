from unittest.mock import Mock

import pytest

from app.storage.providers.local import LocalStorage


def test_missing_stored_object_is_typed(tmp_path):
    with pytest.raises(FileNotFoundError) as caught:
        LocalStorage(tmp_path).get('missing', tmp_path / 'download')
    assert type(caught.value).__name__ == 'StorageNotFoundError'


def test_storage_permission_error_is_safe(tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path)
    source = tmp_path / 'input'
    source.write_text('test')
    monkeypatch.setattr('app.storage.providers.local.os.link', Mock(side_effect=PermissionError('secret details')))
    with pytest.raises(PermissionError) as caught:
        storage.save(source, 'object', 'text/plain')
    assert type(caught.value).__name__ == 'StoragePermissionError'
    assert 'secret' not in str(caught.value)
    assert not storage.exists('object')


def test_publication_cleanup_failure_identifies_object(tmp_path, monkeypatch):
    storage = LocalStorage(tmp_path)
    source = tmp_path / 'input'
    source.write_text('test')
    monkeypatch.setattr('pathlib.Path.unlink', Mock(side_effect=OSError('secret details')))
    with pytest.raises(RuntimeError) as caught:
        storage.save(source, 'object', 'text/plain')
    assert type(caught.value).__name__ == 'StoragePublicationError'
    assert caught.value.storage_key == 'object'
    assert caught.value.completed_steps == ('publish',)
