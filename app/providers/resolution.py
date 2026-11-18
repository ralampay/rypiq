"""Map application configuration and catalog defaults to provider settings."""
from copy import deepcopy
from dataclasses import replace

from app.exceptions import ProviderConfigurationError
from app.models.application import Application, validate_application
from app.providers.catalog import CatalogEntry, resolve_model
from app.providers.config import (
    CAPABILITIES, PROVIDERS, ProviderSettings, DiscoverySettings, validate_provider_config,
)


def settings_for(application: Application, capability: str) -> ProviderSettings:
    if capability not in CAPABILITIES or application.provision_type not in PROVIDERS:
        raise ProviderConfigurationError('Unsupported provider capability')
    model = getattr(application, capability + '_model')
    if not isinstance(model, str) or not model.strip():
        raise ProviderConfigurationError('Application has no configured model for this capability')
    validate_provider_config(application.data, application.embedding_size)
    options = dict(application.data.get('provider_config', {}).get(capability, {}))
    provider = application.provision_type
    if provider == 'local' and capability == 'embedding':
        options.setdefault('document_prefix', 'title: none | text: ')
        options.setdefault('query_prefix', 'task: search result | query: ')
    if provider == 'openai':
        options.setdefault('base_url', 'https://api.openai.com/v1')
        options.setdefault('api_key_env', 'OPENAI_API_KEY')
    if provider == 'llama-server' and 'base_url' not in options:
        raise ProviderConfigurationError('llama-server requires a base URL for the requested capability')
    return ProviderSettings(provider=provider, capability=capability, model=model,
                            embedding_size=application.embedding_size, **options)


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


def discovery_settings_for(application: Application, capability: str) -> DiscoverySettings:
    """Resolve a configured endpoint without requiring a selected model."""
    if capability not in CAPABILITIES or application.provision_type not in ('openai', 'llama-server'):
        raise ProviderConfigurationError('Unsupported discovery provider capability')
    validate_provider_config(application.data, application.embedding_size)
    options = application.data.get('provider_config', {}).get(capability, {})
    provider = application.provision_type
    base_url = options.get('base_url', 'https://api.openai.com/v1' if provider == 'openai' else None)
    if provider == 'llama-server' and not base_url:
        raise ProviderConfigurationError('llama-server requires a base URL for the requested capability')
    return DiscoverySettings(
        provider=provider, base_url=base_url,
        api_key_env=options.get('api_key_env', 'OPENAI_API_KEY' if provider == 'openai' else None),
        timeout=options.get('timeout', 60),
    )
