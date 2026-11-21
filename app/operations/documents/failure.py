"""Record retry diagnostics without replacing the original document failure."""
from uuid import UUID

from app.database.repositories.document import DocumentRepository


def record_failure(repository: DocumentRepository, document_id: UUID, failure: BaseException) -> None:
    # Preserve the triggering exception, even if the database is unavailable.
    failure._rypiq_document_id = document_id
    failure.add_note(f'Document ID: {document_id}. Retry with documents:process --document-id {document_id}.')
    try:
        repository.set_status(document_id, 'error')
    except Exception:
        failure.add_note('Could not record error status; the document may still be pending or processing.')

