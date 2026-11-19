"""Read-only configuration diagnostics at the application composition boundary."""
import os
from pathlib import Path
import re

import yaml

from app.environment import load_environment

from app.database.config import PROJECT_ROOT, validate_config
from app.database.diagnostics import check_database
from app.models.diagnostics import ConfigurationCheck, ConfigurationReport
from app.storage.config import load_storage_config
from app.providers.catalog import DEFAULT_CATALOG, load_catalog


def _section(path: Path, environment: str, name: str) -> tuple[dict | None, ConfigurationCheck]:
    location = f'{path} (environment: {environment})'
    try:
        raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    except FileNotFoundError:
        return None, ConfigurationCheck(name, 'REQUIRED', f'{location}; create this configuration file.')
    except (OSError, UnicodeError, yaml.YAMLError):
        return None, ConfigurationCheck(name, 'INVALID', f'{location}; check file permissions and YAML syntax.')
    if not isinstance(raw, dict) or not isinstance(raw.get(environment), dict):
        return None, ConfigurationCheck(name, 'INVALID', f'{location}; add the selected environment as a YAML mapping.')
    return raw[environment], ConfigurationCheck(name, 'COMPLETE', location)


def _database_checks() -> list[ConfigurationCheck]:
    path = Path(os.getenv('RYPIQ_DATABASE_CONFIG', PROJECT_ROOT / 'config/database.yml')).expanduser().resolve()
    environment = os.getenv('DATABASE_ENV', 'development')
    settings, check = _section(path, environment, 'Database configuration')
    if settings is None:
        return [check]
    missing = [key for key in ('adapter', 'host', 'user', 'database') if key not in settings]
    if missing:
        return [ConfigurationCheck(check.name, 'REQUIRED', f'{check.detail}; add keys: {", ".join(missing)}.')]
    credentials = []
    if 'password_env' in settings:
        variable = settings['password_env']
        if not isinstance(variable, str) or re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', variable) is None:
            return [ConfigurationCheck(check.name, 'INVALID', f'{check.detail}; password_env must name an environment variable.')]
        if variable not in os.environ:
            credentials.append(ConfigurationCheck('Database credentials', 'REQUIRED', f'export {variable} before running RyPiq.'))
            # Validate the remaining settings independently of the absent credential.
            settings = {key: value for key, value in settings.items() if key != 'password_env'}
        else:
            credentials.append(ConfigurationCheck('Database credentials', 'COMPLETE', f'{variable} is set (value hidden).'))
    try:
        settings = validate_config(settings)
    except (ValueError, TypeError):
        check = ConfigurationCheck(check.name, 'INVALID', f'{check.detail}; use adapter postgresql, valid host/user/database, '
                                   'a separate maintenance database, and valid port/dimensions. See config/database.yml.')
    results = [check, *credentials]
    if all(item.status == 'COMPLETE' for item in results):
        results.extend(check_database(settings=settings))
    return results


def _storage_checks() -> list[ConfigurationCheck]:
    path = Path(os.getenv('RYPIQ_STORAGE_CONFIG', PROJECT_ROOT / 'config/storage.yml')).expanduser().resolve()
    environment = os.getenv('STORAGE_ENV', os.getenv('DATABASE_ENV', 'development'))
    settings, check = _section(path, environment, 'Storage configuration')
    if settings is None:
        return [check]
    if settings.get('adapter') != 'local' or not isinstance(settings.get('root'), str) or not settings['root'].strip():
        return [ConfigurationCheck(check.name, 'INVALID', f'{check.detail}; set adapter: local and a nonempty root path.')]
    try:
        # Reuse storage's path resolution, including its config-directory convention.
        root = Path(load_storage_config()['root'])
        existing = root
        while not existing.exists() and existing != existing.parent:
            existing = existing.parent
        if not existing.is_dir():
            detail = f'{root}; the root or an existing ancestor is not a directory. Choose a directory path.'
            status = 'INVALID'
        elif not os.access(existing, os.W_OK | os.X_OK | (os.R_OK if existing == root else 0)):
            detail = f'{root}; check directory permissions on {existing}.'
            status = 'INVALID'
        else:
            detail = f'{root}; ' + ('directory permissions available.' if existing == root else 'directory can be created; parent permissions available.')
            status = 'COMPLETE'
    except (OSError, ValueError, TypeError, KeyError, RuntimeError, yaml.YAMLError):
        status, detail = 'INVALID', 'Cannot inspect storage root; check its path and filesystem permissions.'
    return [check, ConfigurationCheck('Storage path', status, detail)]


def _catalog_checks() -> list[ConfigurationCheck]:
    path = Path(os.getenv('RYPIQ_MODELS_CONFIG', DEFAULT_CATALOG)).expanduser().resolve()
    if not path.exists():
        return [ConfigurationCheck('Model catalog', 'OPTIONAL', f'{path}; absent. Use direct model paths/IDs or create an alias catalog.')]
    try:
        catalog = load_catalog(path)
    except (OSError, ValueError, TypeError, UnicodeError):
        return [ConfigurationCheck('Model catalog', 'OPTIONAL', f'{path}; invalid or unreadable. Check YAML and model options before using aliases.')]
    checks = [ConfigurationCheck('Model catalog', 'COMPLETE', f'{path}; optional aliases configured.')]
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
            if os.getenv(variable):
                checks.append(ConfigurationCheck(f'Credential for @{alias}', 'COMPLETE', f'{variable} is set (value hidden).'))
            else:
                checks.append(ConfigurationCheck(f'Credential for @{alias}', 'OPTIONAL', f'export {variable} before using this alias.'))
    return checks


def check_configuration() -> ConfigurationReport:
    """Inspect local configuration and PostgreSQL readiness without changing data."""
    load_environment()
    checks = []
    for name, inspect in (('Database configuration', _database_checks),
                          ('Storage configuration', _storage_checks),
                          ('Model catalog', _catalog_checks)):
        try:
            checks.extend(inspect())
        except (OSError, ValueError, TypeError, RuntimeError, UnicodeError):
            # Invalid paths and permission failures must not prevent independent checks.
            checks.append(ConfigurationCheck(name, 'OPTIONAL' if name == 'Model catalog' else 'INVALID',
                                             'Cannot inspect configuration; check the configured path and permissions.'))
    return ConfigurationReport(os.getenv('DATABASE_ENV', 'development'), tuple(checks))
