"""Durable embedding attempt; credentials and provider diagnostics are never stored."""
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from uuid import UUID


@dataclass(frozen=True)
class ProcessingJob:
    id: UUID
    document_id: UUID
    batch_size: int
    state: Literal['queued', 'running', 'succeeded', 'failed']
    attempts: int
    created_at: datetime
    updated_at: datetime
    claim_token: UUID | None = None
    lease_expires_at: datetime | None = None
    error: str | None = None
