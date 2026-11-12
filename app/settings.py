"""API configuration resolved after loading the environment."""

from dataclasses import dataclass
import os
from pathlib import Path

from app.operations.embedding import DEFAULT_MODEL


@dataclass(frozen=True)
class Settings:
    inference_model: str
    embedding_model: str

    @classmethod
    def from_environment(cls):
        inference_model = os.getenv("INFERENCE_MODEL", "").strip()
        if not inference_model:
            raise ValueError("INFERENCE_MODEL must point to a local generation GGUF file")
        embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL).strip()
        for name, value in (("INFERENCE_MODEL", inference_model),
                            ("EMBEDDING_MODEL", embedding_model)):
            if not value or not Path(value).expanduser().is_file():
                raise ValueError(f"{name} file does not exist: {value}")
        return cls(
            inference_model=str(Path(inference_model).expanduser().resolve()),
            embedding_model=str(Path(embedding_model).expanduser().resolve()),
        )
