"""Persist a retryable original before any model or extraction work."""
from pathlib import Path

from app.database.repositories.document import DocumentRepository
from app.models.application import Application, validate_application
from app.models.document import Document, DocumentDetails
from app.operations.documents.name import prepare_document_name
from app.operations.documents.process import record_failure
from app.operations.storage.save import save_file
from app.storage.base import ObjectStorage


class CreateDocument:
    def __init__(self, *, name: str, path: str, application: Application,
                 repository: DocumentRepository, storage: ObjectStorage) -> None:
        self.name = name
        self.path = path
        self.application = application
        self.repository = repository
        self.storage = storage

    def execute(self) -> DocumentDetails:
        validate_application(self.application)
        name = prepare_document_name(name=self.name, application_id=self.application.id, repository=self.repository)
        source = Path(self.path)
        # Check readability before creating the durable record.
        with source.open('rb'):
            pass
        document = self.repository.create(Document(name=name, application_id=self.application.id,
            filename=source.name, source=str(source)))
        try:
            stored = save_file(path=source, storage=self.storage)
            document.storage_key = stored.key
            document.content_type = stored.content_type
            document.metadata = {'storage_checksum': stored.checksum}
            try:
                document = self.repository.update(document)
            except Exception as failure:
                try:
                    self.storage.delete(stored.key)
                except Exception:
                    failure.add_note('Could not compensate the stored source object; cleanup is incomplete.')
                raise
        except Exception as failure:
            record_failure(self.repository, document.id, failure)
            raise
        return DocumentDetails(document, [])
