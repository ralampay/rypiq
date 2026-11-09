from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class StoredObject:
    key: str
    size: int
    content_type: str
    checksum: str
    created_at: datetime | None = None
