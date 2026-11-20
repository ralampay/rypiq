"""Immutable configuration snapshots. No resources or implicit reloads live here."""
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TypeVar

from app import environment
from app.database.config import load_database_config, validate_config
from app.exceptions import (ConfigurationError, DatabaseConfigurationError,
                            StorageConfigurationError, ModelCatalogError)
from app.logging.config import LoggingSettings, load_logging_settings, validate_settings
from app.providers.catalog import CatalogEntry, load_catalog
from app.storage.config import load_storage_config


@dataclass(frozen=True)
class ApiSettings:
    secret_key: str = field(default='', repr=False)
    max_upload_bytes: int = 100 * 1024 * 1024
    cors_origins: tuple[str, ...] = ('*',)

    def __post_init__(self):
        object.__setattr__(self, 'cors_origins', tuple(self.cors_origins))


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    user: str
    database: str
    password: str | None = field(default=None, repr=False)
    port: int = 5432
    dimensions: int = 768
    maintenance_database: str = 'postgres'
    adapter: str = 'postgresql'

    def as_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass(frozen=True)
class StorageSettings:
    root: Path
    adapter: str = 'local'

    def __post_init__(self):
        if not isinstance(self.root, (str, Path)) or not str(self.root).strip():
            raise StorageConfigurationError('Local storage requires a nonempty root path')
        object.__setattr__(self, 'root', Path(self.root).expanduser().resolve())

    def as_dict(self) -> dict:
        return {'adapter': self.adapter, 'root': str(self.root)}


@dataclass(frozen=True)
class ConfigurationIssue:
    error_type: type[ConfigurationError]
    message: str
    required: bool = False

    def raise_error(self) -> None:
        raise self.error_type(self.message)


T = TypeVar('T')


@dataclass(frozen=True)
class CredentialValues(Mapping[str, str]):
    """Read-only credential lookup whose representation never reveals values."""
    _values: Mapping[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        object.__setattr__(self, '_values', MappingProxyType(dict(self._values)))

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


@dataclass(frozen=True)
class AppConfig:
    api: ApiSettings | None = field(default_factory=ApiSettings)
    database: DatabaseSettings | None = None
    storage: StorageSettings | None = None
    logging: LoggingSettings | None = field(default_factory=LoggingSettings)
    catalog: Mapping[str, CatalogEntry] | None = None
    credentials: Mapping[str, str] = field(default_factory=dict, repr=False)
    issues: Mapping[str, ConfigurationIssue] = field(default_factory=dict)
    sources: Mapping[str, Path] = field(default_factory=dict)
    environment: str = 'development'
    project_root: Path = field(default_factory=lambda: environment.PROJECT_ROOT)

    def __post_init__(self):
        object.__setattr__(self, 'project_root', Path(self.project_root).resolve())
        object.__setattr__(self, 'credentials', CredentialValues(self.credentials))
        if self.logging is not None:
            try:
                object.__setattr__(self, 'logging', validate_settings(self.logging, project_root=self.project_root))
            except ConfigurationError as error:
                object.__setattr__(self, 'logging', None)
                object.__setattr__(self, 'issues', {**self.issues, 'logging': ConfigurationIssue(type(error), str(error))})
        for name in ('issues', 'sources'):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        if self.catalog is not None:
            object.__setattr__(self, 'catalog', MappingProxyType({name: CatalogEntry(
                entry.alias, entry.provider, tuple(entry.capabilities), entry.model, MappingProxyType(dict(entry.options)))
                for name, entry in self.catalog.items()}))

    def _require(self, name: str, value: T | None, error_type: type[ConfigurationError]) -> T:
        if value is None:
            if name in self.issues:
                self.issues[name].raise_error()
            raise error_type(f'{name.capitalize()} configuration is required')
        return value

    def require_api(self) -> ApiSettings:
        value = self._require('api', self.api, ConfigurationError)
        if not isinstance(value.secret_key, str) or not value.secret_key.strip():
            raise ConfigurationError('APP_SECRET_KEY must be set to a nonempty service secret')
        if type(value.max_upload_bytes) is not int or value.max_upload_bytes < 1:
            raise ConfigurationError('API_MAX_UPLOAD_BYTES must be a positive integer')
        if any(not isinstance(origin, str) or not origin.strip() for origin in value.cors_origins):
            raise ConfigurationError('CORS origins must be nonempty strings')
        return value

    def require_database(self) -> DatabaseSettings:
        value = self._require('database', self.database, DatabaseConfigurationError)
        validate_config(value.as_dict(), environ={})
        return value

    def require_storage(self) -> StorageSettings:
        value = self._require('storage', self.storage, StorageConfigurationError)
        if value.adapter != 'local':
            raise StorageConfigurationError('Only the local storage adapter is supported')
        return value

    def require_logging(self) -> LoggingSettings:
        return self._require('logging', self.logging, ConfigurationError)

    def require_catalog(self) -> dict[str, CatalogEntry]:
        value = self._require('catalog', self.catalog, ModelCatalogError)
        # Consumers can merge/modify options without changing the process snapshot.
        return {name: CatalogEntry(entry.alias, entry.provider, entry.capabilities,
                                   entry.model, dict(entry.options)) for name, entry in value.items()}


def load_config(*, environ: Mapping[str, str] | None = None,
                project_root: str | Path | None = None) -> AppConfig:
    """Read all sources once, deferring section failures until a capability uses them."""
    root = environment.PROJECT_ROOT if project_root is None else Path(project_root)
    root = root.expanduser().resolve()
    try:
        values = environment.read_environment(environ=environ, project_root=root)
    except ConfigurationError as error:
        issue = ConfigurationIssue(type(error), str(error))
        return AppConfig(api=None, logging=None, project_root=root,
                         issues={name: issue for name in ('api', 'database', 'storage', 'logging', 'catalog')})
    issues = {}
    def capture(name, load):
        try:
            return load()
        except ConfigurationError as error:
            issues[name] = ConfigurationIssue(type(error), str(error), 'environment variable is missing' in str(error))
            return None
        except (OSError, UnicodeError, ValueError, RuntimeError):
            error_type = {'database': DatabaseConfigurationError, 'storage': StorageConfigurationError,
                          'catalog': ModelCatalogError}.get(name, ConfigurationError)
            issues[name] = ConfigurationIssue(error_type, 'Cannot read configuration; check paths and permissions')
            return None

    def api():
        try:
            limit = int(values.get('API_MAX_UPLOAD_BYTES', str(100 * 1024 * 1024)))
        except ValueError:
            raise ConfigurationError('API_MAX_UPLOAD_BYTES must be a positive integer') from None
        return ApiSettings(secret_key=values.get('APP_SECRET_KEY', ''), max_upload_bytes=limit,
                           cors_origins=tuple(origin.strip() for origin in values.get('API_CORS_ORIGINS', '*').split(',')
                                              if origin.strip()))

    def database():
        settings = load_database_config(environ=values, project_root=root)
        return DatabaseSettings(**{name: settings[name] for name in DatabaseSettings.__dataclass_fields__ if name in settings})

    def storage():
        settings = load_storage_config(environ=values, project_root=root)
        if settings.get('adapter') != 'local':
            raise StorageConfigurationError('Only the local storage adapter is supported')
        return StorageSettings(root=Path(settings['root']))

    api_settings = capture('api', api)
    database_settings = capture('database', database)
    storage_settings = capture('storage', storage)
    logging_settings = capture('logging', lambda: load_logging_settings(environ=values, project_root=root))
    catalog = capture('catalog', lambda: load_catalog(environ=values, project_root=root))
    sources = {}
    for name, variable, default in (
            ('database', 'RYPIQ_DATABASE_CONFIG', 'config/database.yml'),
            ('storage', 'RYPIQ_STORAGE_CONFIG', 'config/storage.yml'),
            ('logging', 'RYPIQ_LOGGING_CONFIG', 'config/logging.yml'),
            ('catalog', 'RYPIQ_MODELS_CONFIG', 'config/models.yml')):
        try:
            source = Path(values.get(variable, root / default)).expanduser().resolve()
            sources[name] = source
            if name in issues and not source.exists():
                issues[name] = replace(issues[name], required=True)
        except (OSError, ValueError, RuntimeError):
            pass  # The section's captured failure remains available to doctor.
    return AppConfig(api=api_settings, database=database_settings, storage=storage_settings,
                     logging=logging_settings, catalog=catalog, credentials=values, issues=issues,
                     sources=sources, environment=values.get('DATABASE_ENV', 'development'), project_root=root)
