from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from ragworkshop.database.models import Document, SearchResult
from ragworkshop.embeddings.base import validate_vectors
from ragworkshop.processors.chunks import Metadata

class ChromaDocumentRepository:

    def __init__(self, collection):
        self.collection = collection
        self.dimensions: int | None = None

    def ensure_space(self, model_name: str, dimensions: int) -> None:
        raise NotImplementedError('TODO: ChromaDocumentRepository.ensure_space')

    def add(self, documents: list[Document]) -> None:
        raise NotImplementedError('TODO: ChromaDocumentRepository.add')

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError('TODO: ChromaDocumentRepository.delete')

    def search_by_vector(self, embedding: list[float], limit: int=5, filters: Metadata | None=None) -> list[SearchResult]:
        raise NotImplementedError('TODO: ChromaDocumentRepository.search_by_vector')

class ChromaDatabaseBackend:

    def __init__(self, path: Path):
        self.path = Path(path)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(path=str(self.path), settings=ChromaSettings(anonymized_telemetry=False))
        return self._client

    def create_database(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        _ = self.client

    def create_documents_collection(self) -> None:
        self.client.get_or_create_collection('documents', embedding_function=None, configuration={'hnsw': {'space': 'cosine'}})

    def _migrations(self):
        return self.client.get_or_create_collection('ragworkshop_migrations', embedding_function=None)

    def applied_migrations(self) -> set[str]:
        return set(self._migrations().get(include=[])['ids'])

    def record_migration(self, version: str) -> None:
        self._migrations().upsert(ids=[version], embeddings=[[1.0]], documents=[version])

    def documents(self) -> ChromaDocumentRepository:
        return ChromaDocumentRepository(self.client.get_collection('documents', embedding_function=None))
