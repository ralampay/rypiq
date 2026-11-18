"""Public failures must be actionable without parsing diagnostic messages."""
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app import context
from app.models.embedding import validate_embedding
from app.services.application import ApplicationService


def test_missing_application_has_identity_and_compatible_type():
    identity = uuid4()
    repository = Mock()
    repository.get.return_value = None
    with pytest.raises(KeyError) as caught:
        ApplicationService(repository=repository).get(identity)
    assert type(caught.value).__name__ == 'ApplicationNotFoundError'
    assert caught.value.resource_id == identity


def test_invalid_vector_has_distinct_type():
    with pytest.raises(ValueError) as caught:
        validate_embedding([0, 0, 0], 3)
    assert type(caught.value).__name__ == 'InvalidEmbeddingError'


def test_setup_cleanup_does_not_replace_original_failure(monkeypatch):
    failure = ValueError('storage configuration failed')
    backend = Mock()
    backend.close.side_effect = RuntimeError('sensitive cleanup detail')
    monkeypatch.setattr(context, 'get_database', lambda: backend)
    monkeypatch.setattr(context, 'get_storage', Mock(side_effect=failure))
    with pytest.raises(ValueError) as caught:
        context.create_context()
    assert caught.value is failure
    assert 'cleanup' in failure.__notes__[0].lower()
    assert 'sensitive' not in failure.__notes__[0]


@pytest.mark.parametrize('resource', ['document', 'embedding'])
def test_missing_records_are_catchable_by_family(resource):
    from app.exceptions import RyPiqError, NotFoundError, DocumentNotFoundError, EmbeddingNotFoundError
    from app.operations.documents.get import get_document
    from app.operations.embeddings.get import get_embedding
    identity = uuid4()
    repository = Mock()
    repository.get.return_value = repository.details.return_value = None
    action, concrete = ((get_document, DocumentNotFoundError) if resource == 'document'
                        else (get_embedding, EmbeddingNotFoundError))
    with pytest.raises(concrete) as caught:
        action(**{resource + '_id': identity}, repository=repository)
    assert isinstance(caught.value, (KeyError, NotFoundError))
    assert isinstance(caught.value, RyPiqError)
    assert caught.value.resource_id == identity


def test_invalid_identifier_fails_before_repository_access():
    from app.exceptions import InvalidIdentifierError, ValidationError, RyPiqError
    repository = Mock()
    with pytest.raises(InvalidIdentifierError) as caught:
        ApplicationService(repository=repository).get('bad-id')
    assert isinstance(caught.value, ValueError)
    assert isinstance(caught.value, ValidationError)
    assert isinstance(caught.value, RyPiqError)
    repository.get.assert_not_called()


def test_context_exit_preserves_interrupt_and_attempts_cleanup():
    from app.context import ApplicationContext
    backend = Mock()
    backend.close.side_effect = OSError('sensitive cleanup')
    original = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt) as caught:
        with ApplicationContext(documents=Mock(), storage=Mock(), database=backend):
            raise original
    assert caught.value is original
    backend.close.assert_called_once()
    assert 'cleanup' in original.__notes__[0]


def test_context_cleanup_alone_raises_resource_cleanup_error():
    from app.context import ApplicationContext
    from app.exceptions import ResourceCleanupError, RyPiqError
    backend = Mock()
    backend.close.side_effect = OSError('sensitive cleanup')
    with pytest.raises(ResourceCleanupError) as caught:
        with ApplicationContext(documents=Mock(), storage=Mock(), database=backend):
            pass
    assert isinstance(caught.value, RuntimeError)
    assert isinstance(caught.value, RyPiqError)
    assert 'sensitive' not in str(caught.value)


def test_legacy_provider_import_is_same_public_class():
    from app.exceptions import ProviderError, ProviderResponseError, ProviderCleanupError, ResourceCleanupError
    from app.providers.contracts import ProviderError as LegacyProviderError
    assert LegacyProviderError is ProviderError
    assert isinstance(ProviderResponseError('bad response'), ValueError)
    assert isinstance(ProviderResponseError('bad response'), RuntimeError)
    assert isinstance(ProviderCleanupError('cleanup failed'), ResourceCleanupError)


@pytest.mark.parametrize('command, completed', [
    ('documents:create', 'persist'), ('documents:show', 'load'), ('embeddings:show', 'load'),
])
def test_cli_export_failure_keeps_recovery_identity(monkeypatch, tmp_path, capsys, command, completed):
    from app import cli
    from app.exceptions import ExportError, PartialOperationError
    identity = str(uuid4())
    result = {'id': identity}
    monkeypatch.setattr(cli, 'execute', Mock(return_value=result))
    monkeypatch.setattr(cli, 'write_json_export', Mock(side_effect=OSError('sensitive details')))
    arguments = ([command, 'input.txt', '--name', 'report', '--application-id', str(uuid4())]
                 if command == 'documents:create' else [command, '--id', identity])
    with pytest.raises(SystemExit) as caught:
        cli.main(arguments + ['--output', str(tmp_path / 'export.json')])
    error = caught.value.__context__
    assert isinstance(error, ExportError)
    assert isinstance(error, PartialOperationError)
    assert error.resource_id == identity
    assert error.operation == command
    assert error.completed_steps == (completed,)
    assert caught.value.code == 2
    output = capsys.readouterr()
    assert output.out == ''
    assert 'sensitive' not in output.err
