from pathlib import Path

import chromadb

from app.database.base import DatabaseBackend
from app.database.repositories.chroma import (
    ChromaDocumentRepository,
)


class ChromaBackend(DatabaseBackend):

    MIGRATIONS_COLLECTION = "rypiq_schema_migrations"

    def __init__(self, config: dict):
        self.path = Path(config["path"])
        self.collection_name = config.get(
            "collections", {}
        ).get("documents", "documents")

        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = chromadb.PersistentClient(
                path=str(self.path)
            )

        return self._client

    def create_database(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        _ = self.client

    def create_documents_collection(self) -> None:
        self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=None,
            configuration={
                "hnsw": {
                    "space": "cosine",
                }
            },
        )

    def _migrations(self):
        return self.client.get_or_create_collection(
            name=self.MIGRATIONS_COLLECTION,
            embedding_function=None,
        )

    def applied_migrations(self) -> set[str]:
        result = self._migrations().get(
            include=[],
        )
        return set(result["ids"])

    def record_migration(
        self,
        version: str,
        description: str,
    ) -> None:
        self._migrations().add(
            ids=[version],
            documents=[description],
            embeddings=[[1.0]],
        )

    def documents(self) -> ChromaDocumentRepository:
        collection = self.client.get_collection(
            name=self.collection_name,
            embedding_function=None,
        )

        return ChromaDocumentRepository(collection)
