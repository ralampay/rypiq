from __future__ import annotations

import json
from datetime import datetime, timezone
from uuid import UUID

from app.database.repositories.base import (
    DocumentRepository,
)
from app.models.document import Document
from app.models.search import SearchResult


class ChromaDocumentRepository(DocumentRepository):

    def __init__(self, collection):
        self.collection = collection

    def _metadata(self, document: Document) -> dict:
        return {
            "storage_key": document.storage_key or "",
            "filename": document.filename,
            "content_type": document.content_type,
            "source": document.source,
            "source_id": (
                str(document.source_id)
                if document.source_id else ""
            ),
            "chunk_index": document.chunk_index,
            "embedding_model": document.embedding_model,
            "metadata_json": json.dumps(
                document.metadata,
                allow_nan=False,
            ),
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat(),
        }

    def _deserialize(self, result: dict, i: int) -> Document:
        metadata = result["metadatas"][i]
        source_id = metadata.get("source_id", "")
        embedding = result["embeddings"][i]

        return Document(
            id=UUID(result["ids"][i]),
            content=result["documents"][i],
            embedding=embedding.tolist(),
            storage_key=(
                metadata.get("storage_key") or None
            ),
            filename=metadata.get("filename", ""),
            content_type=metadata.get("content_type", ""),
            source=metadata.get("source", ""),
            source_id=UUID(source_id) if source_id else None,
            chunk_index=metadata["chunk_index"],
            embedding_model=metadata["embedding_model"],
            metadata=json.loads(
                metadata.get("metadata_json", "{}")
            ),
            created_at=datetime.fromisoformat(
                metadata["created_at"]
            ),
            updated_at=datetime.fromisoformat(
                metadata["updated_at"]
            ),
        )

    def create(self, document: Document) -> Document:
        self.collection.add(
            ids=[str(document.id)],
            documents=[document.content],
            embeddings=[document.embedding],
            metadatas=[self._metadata(document)],
        )
        return document

    def get(self, id: UUID) -> Document | None:
        result = self.collection.get(
            ids=[str(id)],
            include=[
                "documents",
                "embeddings",
                "metadatas",
            ],
        )

        if not result["ids"]:
            return None

        return self._deserialize(result, 0)

    def update(self, document: Document) -> Document:
        if self.get(document.id) is None:
            raise KeyError(document.id)

        document.updated_at = datetime.now(timezone.utc)

        self.collection.update(
            ids=[str(document.id)],
            documents=[document.content],
            embeddings=[document.embedding],
            metadatas=[self._metadata(document)],
        )
        return document

    def delete(self, id: UUID) -> bool:
        if self.get(id) is None:
            return False

        self.collection.delete(ids=[str(id)])
        return True

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Document]:
        if limit < 1 or offset < 0:
            raise ValueError("Invalid pagination")

        result = self.collection.get(
            limit=limit,
            offset=offset,
            include=[
                "documents",
                "embeddings",
                "metadatas",
            ],
        )

        return [
            self._deserialize(result, i)
            for i in range(len(result["ids"]))
        ]

    def search_by_vector(
        self,
        embedding: list[float],
        limit: int = 10,
        filters: dict | None = None,
    ) -> list[SearchResult]:

        if not embedding:
            raise ValueError("Embedding cannot be empty")

        if limit < 1:
            raise ValueError("Limit must be positive")

        kwargs = {
            "query_embeddings": [embedding],
            "n_results": limit,
        }

        if filters is not None:
            kwargs["where"] = filters

        result = self.collection.query(
            **kwargs,
            include=[
                "documents",
                "metadatas",
                "embeddings",
                "distances",
            ],
        )

        matches = []

        for index, record_id in enumerate(result["ids"][0]):

            document_data = {
                "ids": [record_id],
                "documents": [
                    result["documents"][0][index]
                ],
                "metadatas": [
                    result["metadatas"][0][index]
                ],
                "embeddings": [
                    result["embeddings"][0][index]
                ],
            }

            document = self._deserialize(
                document_data,
                0,
            )

            matches.append(
                SearchResult(
                    document=document,
                    distance=float(
                        result["distances"][0][index]
                    ),
                )
            )

        return matches
