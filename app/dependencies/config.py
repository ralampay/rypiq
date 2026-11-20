"""The API instance owns its configuration; dependencies borrow that snapshot."""
from typing import Annotated
from fastapi import Depends, Request
from app.config import AppConfig


def get_config(request: Request) -> AppConfig:
    return request.app.state.config


ConfigDependency = Annotated[AppConfig, Depends(get_config)]
