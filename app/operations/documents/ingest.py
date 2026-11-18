from app.exceptions import ModelMismatchError
from app.models.application import Application, validate_application
from app.providers.config import embedding_preprocessing, model_identity
from app.database.repositories.document import DocumentRepository
from app.storage.base import ObjectStorage
from app.models.document import DocumentDetails
from app.providers.contracts.embedding import EmbeddingProvider
from app.operations.embeddings.embed_file import EmbedTextFile
from app.operations.documents.persist import persist_document
from app.operations.documents.name import prepare_document_name


class IngestDocument:
    def __init__(self, *, name: str, path: str, storage: ObjectStorage,
                 repository: DocumentRepository, embedding_provider: EmbeddingProvider,
                 chunk_size: int, stride: int, application: Application, batch_size: int = 1) -> None:
        self.name = name
        self.application = application
        self.path = path
        self.storage = storage
        self.repository = repository
        self.provider = embedding_provider
        self.chunk_size = chunk_size
        self.stride = stride
        self.batch_size = batch_size

    def execute(self) -> DocumentDetails:
        validate_application(self.application)
        if self.provider.model_name != model_identity(self.application.embedding_model, self.application.provision_type):
            raise ModelMismatchError("Embedding provider model must match the application")
        name = prepare_document_name(name=self.name, application_id=self.application.id, repository=self.repository)
        result = EmbedTextFile(path=self.path, chunk_size=self.chunk_size,
                               stride=self.stride, embedding_provider=self.provider,
                               batch_size=self.batch_size,
                               document_prefix=embedding_preprocessing(self.application.data, self.application.provision_type, self.application.embedding_size)[0]).execute()
        return persist_document(self.path, self.storage, self.repository,
                                self.chunk_size, self.stride, result, self.application, name=name)
