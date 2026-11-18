"""Normalized catalog and discovery values."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    id: str
    provider: str
    capabilities: tuple[str, ...]
    source: str
    availability: str
    model: str


