from dataclasses import replace
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.models.application import Application
from app.models.document import Document
from app.services.application import ApplicationService
from app.services.document import DocumentService


@pytest.fixture
def application():
    return Application(name='test', embedding_model='model', embedding_size=2)


@pytest.fixture
def document_service(application):
    repository, applications = Mock(), Mock()
    applications.get.return_value = application
    service = DocumentService(application_id=application.id, repository=repository, applications=applications)
    assert not repository.mock_calls and not applications.mock_calls
    return service


def test_application_crud_borrows_repository(application):
    repository = Mock()
    service = ApplicationService(repository=repository)
    assert not repository.mock_calls
    repository.create.return_value = application
    repository.get.return_value = application
    repository.update.return_value = application
    repository.list.return_value = [application]
    repository.delete.side_effect = [True, False]
    assert service.create(application) is application
    assert service.get(application.id) is application
    assert service.list(limit=2, offset=1) == [application]
    assert service.update(application) is application
    assert service.delete(application.id) is True
    assert service.delete(application.id) is False
    repository.list.assert_called_once_with(limit=2, offset=1)
    repository.close.assert_not_called()


def test_application_missing_and_failure_propagation(application):
    repository = Mock()
    service = ApplicationService(repository=repository)
    repository.get.return_value = None
    with pytest.raises(KeyError):
        service.get(application.id)
    failure = RuntimeError('failure')
    repository.update.side_effect = failure
    with pytest.raises(RuntimeError) as caught:
        service.update(application)
    assert caught.value is failure
    repository.close.assert_not_called()


@pytest.mark.parametrize('method', ['create', 'update'])
def test_application_invalid_write_before_io(application, method):
    repository = Mock()
    service = ApplicationService(repository=repository)
    with pytest.raises(ValueError):
        getattr(service, method)(replace(application, name=''))
    assert not repository.mock_calls


@pytest.mark.parametrize('method', ['get', 'delete'])
def test_application_invalid_id_before_io(method):
    repository = Mock()
    with pytest.raises(ValueError, match='UUID'):
        getattr(ApplicationService(repository=repository), method)('not-a-uuid')
    assert not repository.mock_calls


@pytest.mark.parametrize('limit,offset', [(0, 0), (True, 0), (1, -1), (1, False)])
def test_service_pagination_before_io(document_service, limit, offset):
    for service in (document_service, ApplicationService(repository=Mock())):
        with pytest.raises(ValueError):
            service.list(limit=limit, offset=offset)
        assert not service.repository.mock_calls
    assert not document_service.applications.mock_calls


def test_document_create_snapshots_application_without_mutating_input(document_service, application):
    original = Document(filename='file.txt')
    document_service.repository.create.side_effect = lambda document: document
    created = document_service.create(original)
    assert created is not original
    assert original.application_id is None
    assert created.application_id == application.id
    assert created.filename == 'file.txt'
    document_service.repository.close.assert_not_called()
    document_service.applications.close.assert_not_called()
    # An already assigned document in this application is also accepted.
    assert document_service.create(created).application_id == application.id


def test_document_scoped_read_update_delete_and_list(document_service, application):
    document = Document(application_id=application.id,
                        filename='file.txt')
    repository = document_service.repository
    repository.get.return_value = document
    repository.list.return_value = [document]
    repository.update.return_value = document
    repository.delete.side_effect = [True, False]
    assert document_service.get(document.id) is document
    assert document_service.list(limit=2, offset=1) == [document]
    assert document_service.update(document) is document
    assert document_service.delete(document.id) is True
    assert document_service.delete(document.id) is False
    repository.get.assert_called_once_with(document.id, application_id=application.id)
    repository.list.assert_called_once_with(limit=2, offset=1, application_id=application.id)
    repository.update.assert_called_once_with(document, application_id=application.id)
    assert repository.delete.call_args.kwargs == {'application_id': application.id}
    repository.close.assert_not_called()


def test_document_missing_resources(document_service):
    document_service.repository.get.return_value = None
    with pytest.raises(KeyError):
        document_service.get(uuid4())
    document_service.applications.get.return_value = None
    with pytest.raises(KeyError):
        document_service.create(Document())
    document_service.repository.create.assert_not_called()


@pytest.mark.parametrize('method', ['create', 'update'])
def test_document_cross_application_write_rejected(document_service, method):
    with pytest.raises(ValueError, match='application'):
        getattr(document_service, method)(Document(application_id=uuid4()))
    assert not document_service.repository.mock_calls


@pytest.mark.parametrize('method', ['get', 'delete'])
def test_document_invalid_id_rejected(document_service, method):
    with pytest.raises(ValueError, match='UUID'):
        getattr(document_service, method)('invalid')
    assert not document_service.repository.mock_calls


@pytest.mark.parametrize('method', ['create', 'update', 'get', 'delete', 'list'])
def test_document_invalid_scope_rejected(method):
    service = DocumentService(application_id=None, repository=Mock(), applications=Mock())
    args = [] if method == 'list' else [Document()] if method in ('create', 'update') else [uuid4()]
    with pytest.raises(ValueError, match='UUID'):
        getattr(service, method)(*args)
    assert not service.repository.mock_calls and not service.applications.mock_calls


@pytest.mark.parametrize('changes', [
    {'filename': None}, {'metadata': []}, {'metadata': {'bad': float('nan')}},
    {'storage_key': 5}, {'id': 'invalid'},
])
@pytest.mark.parametrize('method', ['create', 'update'])
def test_document_invalid_values_never_write(document_service, application, method, changes):
    document = Document(application_id=application.id,
                        filename='file.txt')
    with pytest.raises(ValueError):
        getattr(document_service, method)(replace(document, **changes))
    document_service.repository.create.assert_not_called()
    document_service.repository.update.assert_not_called()


@pytest.mark.parametrize('method', ['create', 'get', 'list', 'update', 'delete'])
def test_document_repository_failure_preserved_and_resources_borrowed(document_service, application, method):
    document = Document(application_id=application.id,
                        filename='file.txt')
    failure = RuntimeError('persistence failed')
    getattr(document_service.repository, method).side_effect = failure
    args = [] if method == 'list' else [document] if method in ('create', 'update') else [document.id]
    with pytest.raises(RuntimeError) as caught:
        getattr(document_service, method)(*args)
    assert caught.value is failure
    document_service.repository.close.assert_not_called()
    document_service.applications.close.assert_not_called()
