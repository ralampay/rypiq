"""HTTP interface to the existing retrieval-augmented inference operation."""

import logging
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, ConfigDict, StringConstraints

from app.operations.inference import infer

logger = logging.getLogger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, strict=True)]


class SourceDocument(BaseModel):
    citation: int
    id: str
    distance: float
    content: str
    storage_key: str | None
    filename: str
    chunk_index: int


class QueryResponse(BaseModel):
    response: str
    documents: list[SourceDocument]


@router.post("/query", response_model=QueryResponse, tags=["inference"])
def query_documents(
    body: QueryRequest,
    request: Request,
    top_k: Annotated[int, Query(ge=1, description="Number of document chunks to retrieve")] = 3,
):
    state = request.app.state
    try:
        with state.inference_lock:
            return infer(
                query=body.query,
                model_path=state.settings.inference_model,
                embedding_provider=state.embedding_provider,
                repository=state.context.documents,
                top_k=top_k,
            )
    except ValueError as exc:
        logger.warning("Query could not be processed", exc_info=True)
        raise HTTPException(
            status_code=400,
            detail="Unable to process query. Try a smaller top_k or shorter query.",
        ) from exc
    except (OSError, RuntimeError, KeyError) as exc:
        logger.exception("Local inference failed")
        raise HTTPException(status_code=503, detail="Local inference is unavailable.") from exc
