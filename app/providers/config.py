"""Pure provider settings; environment and filesystem access stay in factories."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.application import Application

from dataclasses import dataclass
import math
import re
from urllib.parse import urlsplit

from app.exceptions import ProviderConfigurationError


PROVIDERS = ('local', 'openai', 'llama-server')
CAPABILITIES = ('embedding', 'inference')
COMMON = {'base_url', 'api_key_env', 'timeout', 'context_size', 'gpu_layers'}
OPTIONS = {
    'embedding': COMMON | {'document_prefix', 'query_prefix', 'dimensions'},
    'inference': COMMON | {'chat_format'},
}


def validate_options(options: dict, capability: str) -> None:
    if not isinstance(options, dict) or set(options) - OPTIONS[capability]:
        raise ProviderConfigurationError('Unknown or invalid provider options; use environment references for credentials')
    for name, value in options.items():
        if name == 'base_url':
            try:
                url = urlsplit(value)
                valid = (url.scheme in ('http', 'https') and bool(url.hostname) and
                         not url.username and not url.password and not url.query and not url.fragment)
                url.port
            except (TypeError, ValueError, AttributeError):
                valid = False
            if not valid:
                raise ProviderConfigurationError('Provider base URL must be HTTP(S) without credentials, query or fragment')
        elif name == 'api_key_env':
            if not isinstance(value, str) or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', value) is None:
                raise ProviderConfigurationError('Credential environment variable name is invalid')
        elif name == 'timeout':
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ProviderConfigurationError('Provider timeout must be positive and finite')
        elif name in ('context_size', 'dimensions', 'gpu_layers'):
            minimum = -1 if name == 'gpu_layers' else 1
            if type(value) is not int or value < minimum or (name == 'dimensions' and value > 16000):
                raise ProviderConfigurationError('Provider numeric option is invalid')
        elif not isinstance(value, str):
            raise ProviderConfigurationError('Provider text option must be a string')


def validate_provider_config(data: dict, embedding_size: int) -> None:
    if not isinstance(data, dict):
        raise ProviderConfigurationError('Application data must be a JSON object')
    config = data.get('provider_config', {})
    if not isinstance(config, dict) or set(config) - set(CAPABILITIES):
        raise ProviderConfigurationError('provider_config must contain embedding and/or inference sections')
    for capability, options in config.items():
        validate_options(options, capability)
    size = config.get('embedding', {}).get('dimensions', embedding_size)
    if size != embedding_size:
        raise ProviderConfigurationError('Requested dimensions must match application embedding size')


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    capability: str
    model: str
    embedding_size: int
    base_url: str | None = None
    api_key_env: str | None = None
    timeout: float = 60
    context_size: int = 2048
    gpu_layers: int = 0
    chat_format: str | None = None
    document_prefix: str = ''
    query_prefix: str = ''
    dimensions: int | None = None


def settings_for(application: Application, capability: str) -> ProviderSettings:
    """Compatibility entrypoint for application settings resolution."""
    from app.providers.resolution import settings_for as resolve
    return resolve(application, capability)


def embedding_preprocessing(data: dict, provider: str, size: int) -> tuple[str, str, int | None]:
    options = data.get('provider_config', {}).get('embedding', {})
    local = provider == 'local'
    return (options.get('document_prefix', 'title: none | text: ' if local else ''),
            options.get('query_prefix', 'task: search result | query: ' if local else ''),
            options.get('dimensions'))


def model_identity(model: str, provider: str) -> str:
    from pathlib import Path
    return Path(model).expanduser().resolve().name if provider == 'local' else model


@dataclass(frozen=True)
class DiscoverySettings:
    """Endpoint configuration independent of any selected model."""
    provider: str
    base_url: str | None = None
    api_key_env: str | None = None
    timeout: float = 60
