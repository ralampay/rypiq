from contextlib import nullcontext
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.database.repositories.postgresql import document as module
from app.database.schema import build_metadata


@pytest.mark.parametrize('stage', ['scalar', 'commit'])
def test_failed_lock_acquisition_discards_connection(monkeypatch, stage):
    connection = Mock()
    getattr(connection, stage).side_effect = RuntimeError('connection failed')
    monkeypatch.setattr(module, 'database_connection', lambda engine: nullcontext(connection))
    repository = module.PostgreSQLDocumentRepository(Mock(), build_metadata().tables['public.documents'])
    with pytest.raises(RuntimeError, match='connection failed'):
        with repository.processing_lock(uuid4()):
            pytest.fail('Acquisition failed')
    connection.invalidate.assert_called_once()
