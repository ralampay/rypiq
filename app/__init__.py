"""RyPiq library; HTTP dependencies are loaded only when creating the API."""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import FastAPI
    from app.config import AppConfig


def create_app(config: AppConfig | None = None) -> FastAPI:
    from app.api import create_app as factory
    return factory(config=config)
