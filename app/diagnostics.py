"""Read-only readiness checks against a captured configuration snapshot."""
import os
from pathlib import Path

from app.config import AppConfig, load_config
from app.database.diagnostics import check_database
from app.exceptions import ConfigurationError
from app.models.diagnostics import ConfigurationCheck, ConfigurationReport


def _location(config: AppConfig, section: str, environment: str) -> str:
    return f'{config.sources.get(section, "injected settings")} (environment: {environment})'


def _failure(config: AppConfig, section: str, name: str, error: ConfigurationError) -> ConfigurationCheck:
    issue = config.issues.get(section)
    status = 'REQUIRED' if issue is not None and issue.required else 'INVALID'
    if section == 'catalog':
        status = 'OPTIONAL'
    selected = config.credentials.get('STORAGE_ENV', config.environment) if section == 'storage' else config.environment
    detail = f'{_location(config, section, selected)}; {error}'
    if section == 'database':
        detail += '; required settings include adapter, host, user, database, and a valid port.'
    return ConfigurationCheck(name, status, detail)


def _database_checks(config: AppConfig) -> list[ConfigurationCheck]:
    try:
        settings = config.require_database()
    except ConfigurationError as error:
        return [_failure(config, 'database', 'Database configuration', error)]
    return [ConfigurationCheck('Database configuration', 'COMPLETE',
                               _location(config, 'database', config.environment)),
            *check_database(settings=settings.as_dict())]


def _storage_checks(config: AppConfig) -> list[ConfigurationCheck]:
    try:
        root = config.require_storage().root
    except ConfigurationError as error:
        return [_failure(config, 'storage', 'Storage configuration', error)]
    check = ConfigurationCheck('Storage configuration', 'COMPLETE', _location(
        config, 'storage', config.credentials.get('STORAGE_ENV', config.environment)))
    try:
        existing = root
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        if not existing.is_dir():
            status, detail = 'INVALID', f'{root}; choose a directory path.'
        elif not os.access(existing, os.W_OK | os.X_OK | (os.R_OK if existing == root else 0)):
            status, detail = 'INVALID', f'{root}; check directory permissions on {existing}.'
        else:
            status = 'COMPLETE'
            detail = f'{root}; ' + ('directory permissions available.' if existing == root
                                   else 'directory can be created; parent permissions available.')
    except (OSError, ValueError, RuntimeError):
        status, detail = 'INVALID', 'Cannot inspect storage root; check its path and filesystem permissions.'
    return [check, ConfigurationCheck('Storage path', status, detail)]


def _catalog_checks(config: AppConfig) -> list[ConfigurationCheck]:
    try:
        catalog = config.require_catalog()
    except ConfigurationError as error:
        return [_failure(config, 'catalog', 'Model catalog', error)]
    checks = [ConfigurationCheck('Model catalog', 'COMPLETE',
                                 f'{config.sources.get("catalog", "injected catalog")}; optional aliases configured.')]
    for alias, entry in catalog.items():
        if entry.provider == 'local':
            model = Path(entry.model)
            if model.is_file() and os.access(model, os.R_OK):
                checks.append(ConfigurationCheck(f'Model @{alias}', 'COMPLETE', f'{model}; readable file, model compatibility not verified.'))
            else:
                checks.append(ConfigurationCheck(f'Model @{alias}', 'OPTIONAL', f'{model}; supply a readable model file before using this alias.'))
        else:
            checks.append(ConfigurationCheck(f'Model @{alias}', 'OPTIONAL', 'Remote availability is not checked; verify saved application provider settings.'))
        variable = entry.options.get('api_key_env', 'OPENAI_API_KEY' if entry.provider == 'openai' else None)
        if variable:
            if config.credentials.get(variable):
                checks.append(ConfigurationCheck(f'Credential for @{alias}', 'COMPLETE', f'{variable} is set (value hidden).'))
            else:
                checks.append(ConfigurationCheck(f'Credential for @{alias}', 'OPTIONAL', f'export {variable} before running RyPiq.'))
    return checks


def check_configuration(*, config: AppConfig | None = None) -> ConfigurationReport:
    """Inspect current readiness without reloading settings or changing data."""
    config = load_config() if config is None else config
    checks = []
    for name, inspect in (('Database configuration', _database_checks),
                          ('Storage configuration', _storage_checks),
                          ('Model catalog', _catalog_checks)):
        try:
            checks.extend(inspect(config))
        except (OSError, ValueError, TypeError, RuntimeError, UnicodeError):
            checks.append(ConfigurationCheck(name, 'OPTIONAL' if name == 'Model catalog' else 'INVALID',
                                             'Cannot inspect configuration; check the configured path and permissions.'))
    return ConfigurationReport(config.environment, tuple(checks))
