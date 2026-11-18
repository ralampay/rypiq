"""Model listing without selecting or loading a model.

Return normalized ModelInfo values; do not infer undeclared capabilities.
Owners close resources; cleanup need not be idempotent.
"""
from typing import Protocol
from app.providers.types.model import ModelInfo


class ModelDiscovery(Protocol):
    def list_models(self) -> list[ModelInfo]: ...
    def close(self) -> None: ...
