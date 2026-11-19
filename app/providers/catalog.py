"""Explicit repository catalog. Alias references use @name; direct models remain valid."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.application import Application

from copy import deepcopy
from dataclasses import dataclass
import os
from pathlib import Path

import yaml

from app.environment import load_environment

from app.exceptions import ModelCatalogError
from app.providers.config import CAPABILITIES, PROVIDERS, validate_options
from app.providers.types.model import ModelInfo

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / 'config/models.yml'


@dataclass(frozen=True)
class CatalogEntry:
    alias: str
    provider: str
    capabilities: tuple[str, ...]
    model: str
    options: dict


def load_catalog(path: str | Path | None = None) -> dict[str, CatalogEntry]:
    load_environment()
    source = Path(path or os.getenv('RYPIQ_MODELS_CONFIG', DEFAULT_CATALOG)).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text())
    except (OSError, UnicodeError, yaml.YAMLError):
        raise ModelCatalogError('Cannot read model catalog; check the configured file') from None
    if not isinstance(raw, dict) or set(raw) != {'models'} or not isinstance(raw['models'], dict):
        raise ModelCatalogError('Model catalog must contain a models mapping')
    result = {}
    for alias, item in raw['models'].items():
        if (not isinstance(alias, str) or not alias.strip() or alias.startswith('@') or
                not isinstance(item, dict) or set(item) - {'provider', 'capabilities', 'path', 'model', 'options'}):
            raise ModelCatalogError('Invalid model catalog entry')
        provider, capabilities = item.get('provider'), item.get('capabilities')
        if (provider not in PROVIDERS or not isinstance(capabilities, list) or not capabilities or
                any(c not in CAPABILITIES for c in capabilities) or len(set(capabilities)) != len(capabilities)):
            raise ModelCatalogError('Invalid catalog provider or capabilities')
        model = item.get('path' if provider == 'local' else 'model')
        if not isinstance(model, str) or not model.strip():
            raise ModelCatalogError('Catalog entry requires a local path or remote model ID')
        options = item.get('options', {})
        for capability in capabilities:
            validate_options(options, capability)
        if provider == 'local':
            model = str((source.parent / Path(model).expanduser()).resolve())
        result[alias] = CatalogEntry(alias, provider, tuple(sorted(capabilities)), model, options)
    return result


def resolve_model(model: str, provider: str, capability: str,
                  catalog: dict[str, CatalogEntry]) -> tuple[str, dict]:
    if not model.startswith('@'):
        return model, {}
    entry = catalog.get(model[1:])
    if entry is None or entry.provider != provider or capability not in entry.capabilities:
        raise ModelCatalogError('Model alias is missing or incompatible with provider/capability')
    return entry.model, deepcopy(entry.options)


def resolve_application(application: Application, catalog: dict[str, CatalogEntry]) -> Application:
    """Compatibility entrypoint for application catalog resolution."""
    from app.providers.resolution import resolve_application as resolve
    return resolve(application, catalog)


def catalog_models(catalog: dict[str, CatalogEntry]) -> list[ModelInfo]:
    return [ModelInfo('@' + entry.alias, entry.provider, entry.capabilities, 'catalog',
                      ('available' if Path(entry.model).is_file() else 'missing')
                      if entry.provider == 'local' else 'unverified', entry.model)
            for entry in sorted(catalog.values(), key=lambda entry: entry.alias)]
