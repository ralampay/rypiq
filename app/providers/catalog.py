"""Explicit repository catalog. Alias references use @name; direct models remain valid."""
from copy import deepcopy
from dataclasses import dataclass, replace
import os
from pathlib import Path

import yaml

from app.models.application import Application, validate_application
from app.providers.config import CAPABILITIES, PROVIDERS, validate_options, validate_provider_config
from app.providers.contracts import ModelInfo

DEFAULT_CATALOG = Path(__file__).resolve().parents[2] / 'config/models.yml'


@dataclass(frozen=True)
class CatalogEntry:
    alias: str
    provider: str
    capabilities: tuple[str, ...]
    model: str
    options: dict


def load_catalog(path: str | Path | None = None) -> dict[str, CatalogEntry]:
    source = Path(path or os.getenv('RYPIQ_MODELS_CONFIG', DEFAULT_CATALOG)).expanduser().resolve()
    try:
        raw = yaml.safe_load(source.read_text())
    except (OSError, yaml.YAMLError):
        raise ValueError('Cannot read model catalog; check the configured file') from None
    if not isinstance(raw, dict) or set(raw) != {'models'} or not isinstance(raw['models'], dict):
        raise ValueError('Model catalog must contain a models mapping')
    result = {}
    for alias, item in raw['models'].items():
        if (not isinstance(alias, str) or not alias.strip() or alias.startswith('@') or
                not isinstance(item, dict) or set(item) - {'provider', 'capabilities', 'path', 'model', 'options'}):
            raise ValueError('Invalid model catalog entry')
        provider, capabilities = item.get('provider'), item.get('capabilities')
        if (provider not in PROVIDERS or not isinstance(capabilities, list) or not capabilities or
                any(c not in CAPABILITIES for c in capabilities) or len(set(capabilities)) != len(capabilities)):
            raise ValueError('Invalid catalog provider or capabilities')
        model = item.get('path' if provider == 'local' else 'model')
        if not isinstance(model, str) or not model.strip():
            raise ValueError('Catalog entry requires a local path or remote model ID')
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
        raise ValueError('Model alias is missing or incompatible with provider/capability')
    return entry.model, deepcopy(entry.options)


def resolve_application(application: Application, catalog: dict[str, CatalogEntry]) -> Application:
    validate_application(application)
    data = deepcopy(application.data)
    validate_provider_config(data, application.embedding_size)
    models = {}
    for capability in CAPABILITIES:
        model = getattr(application, capability + '_model')
        if model is None:
            continue
        resolved, defaults = resolve_model(model, application.provision_type, capability, catalog)
        models[capability + '_model'] = resolved
        if defaults:
            section = data.setdefault('provider_config', {}).setdefault(capability, {})
            data['provider_config'][capability] = {**defaults, **section}
    result = replace(application, data=data, **models)
    validate_provider_config(result.data, result.embedding_size)
    return result


def catalog_models(catalog: dict[str, CatalogEntry]) -> list[ModelInfo]:
    return [ModelInfo('@' + entry.alias, entry.provider, entry.capabilities, 'catalog',
                      ('available' if Path(entry.model).is_file() else 'missing')
                      if entry.provider == 'local' else 'unverified', entry.model)
            for entry in sorted(catalog.values(), key=lambda entry: entry.alias)]
