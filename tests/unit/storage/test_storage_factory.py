import pytest

from app.storage.factory import get_storage
from app.storage.providers.local import LocalStorage


@pytest.mark.unit
def test_local_storage_factory(tmp_path):
    storage = get_storage({
        "adapter": "local",
        "root": str(tmp_path),
    })

    assert isinstance(storage, LocalStorage)


@pytest.mark.unit
def test_unsupported_storage_adapter():
    with pytest.raises(ValueError):
        get_storage({
            "adapter": "unknown",
        })
