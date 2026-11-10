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
        metadata = dict(self.collection.metadata or {})
        if "embedding_space" in metadata:
            if (metadata["embedding_space"] != model_name
                    or metadata["dimensions"] != dimensions):
                raise ValueError("Embedding space mismatch; use a new collection or rebuild")
        else:
            if self.collection.count():
                raise ValueError("Existing collection lacks embedding space metadata")
            self.collection.modify(metadata={**metadata, "embedding_space": model_name,
                                             "dimensions": dimensions})
        self.dimensions = dimensions

    def add(self, documents: list[Document]) -> None:
        if not documents:
            return
        if self.dimensions is None:
            raise ValueError("Call ensure_space before indexing")
        validate_vectors([document.embedding for document in documents], len(documents), self.dimensions)
        for document in documents:
            # Scalar metadata remains filterable; reserved fields cannot be overridden.
            metadata = {**document.metadata, "source_id": document.source_id,
                        "storage_key": document.storage_key}
            self.collection.add(ids=[document.id], documents=[document.content],
                                embeddings=[document.embedding], metadatas=[metadata])

    def delete(self, ids: list[str]) -> None:
        if ids:
            self.collection.delete(ids=ids)

    def search_by_vector(self, embedding: list[float], limit: int = 5,
                         filters: Metadata | None = None) -> list[SearchResult]:
        if self.dimensions is None:
            raise ValueError("Call ensure_space before search")
        validate_vectors([embedding], 1, self.dimensions)
        if type(limit) is not int or limit < 1:
            raise ValueError("limit must be a positive integer")
        count = self.collection.count()
        if not count:
            return []
        where = None
        if filters:
            conditions = [{key: value} for key, value in filters.items()]
            where = conditions[0] if len(conditions) == 1 else {"$and": conditions}
        result = self.collection.query(query_embeddings=[embedding], n_results=min(count, limit),
                                       where=where, include=["documents", "embeddings", "metadatas", "distances"])
        matches = []
        for i, record_id in enumerate(result["ids"][0]):
            metadata = dict(result["metadatas"][0][i])
            document = Document(record_id, metadata.pop("source_id"), result["documents"][0][i],
                                list(map(float, result["embeddings"][0][i])),
                                metadata.pop("storage_key"), metadata)
            matches.append(SearchResult(document, float(result["distances"][0][i])))
        return matches


class ChromaDatabaseBackend:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.path), settings=ChromaSettings(anonymized_telemetry=False))
        return self._client

    def create_database(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        _ = self.client

    def create_documents_collection(self) -> None:
        self.client.get_or_create_collection("documents", embedding_function=None,
                                            configuration={"hnsw": {"space": "cosine"}})

    def _migrations(self):
        return self.client.get_or_create_collection("ragworkshop_migrations", embedding_function=None)

    def applied_migrations(self) -> set[str]:
        return set(self._migrations().get(include=[])["ids"])

    def record_migration(self, version: str) -> None:
        self._migrations().upsert(ids=[version], embeddings=[[1.0]], documents=[version])

    def documents(self) -> ChromaDocumentRepository:
        return ChromaDocumentRepository(self.client.get_collection("documents", embedding_function=None))
