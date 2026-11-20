"""Resolve logging settings at application boundaries."""
from dataclasses import dataclass, replace
import os
from pathlib import Path
import re

import yaml

from app import environment as application_environment
from app.exceptions import ConfigurationError


@dataclass(frozen=True)
class LoggingSettings:
    environment: str = 'development'
    directory: Path = Path('logs')
    level: str = 'INFO'
    format: str = 'text'
    console: bool = False


def validate_settings(settings: LoggingSettings, *, project_root: Path | None = None) -> LoggingSettings:
    if not isinstance(settings.environment, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', settings.environment):
        raise ConfigurationError('Invalid logging environment')
    if not isinstance(settings.level, str) or settings.level.upper() not in {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}:
        raise ConfigurationError('Invalid logging level')
    if settings.format not in ('text', 'json') or type(settings.console) is not bool:
        raise ConfigurationError('Invalid logging format or console setting')
    if not isinstance(settings.directory, (str, Path)) or not str(settings.directory).strip() or '\x00' in str(settings.directory):
        raise ConfigurationError('Invalid logging directory')
    directory = Path(settings.directory).expanduser()
    if not directory.is_absolute():
        directory = (application_environment.PROJECT_ROOT if project_root is None else project_root) / directory
    return replace(settings, directory=directory.resolve(), level=settings.level.upper())


def load_logging_settings(*, environment: str | None = None, config_path: str | Path | None = None,
                          directory: str | Path | None = None, level: str | None = None,
                          format: str | None = None, console: bool | None = None,
                          environ=None, project_root: Path | None = None) -> LoggingSettings:
    if environ is None:
        application_environment.load_environment()
    env = os.environ if environ is None else environ
    root = application_environment.PROJECT_ROOT if project_root is None else project_root
    selected = environment if environment is not None else env.get('LOG_ENV', env.get('DATABASE_ENV', 'development'))
    validate_settings(LoggingSettings(environment=selected), project_root=root)
    explicit_path = config_path if config_path is not None else env.get('RYPIQ_LOGGING_CONFIG')
    path = Path(explicit_path) if explicit_path is not None else root / 'config/logging.yml'
    try:
        with path.expanduser().open(encoding='utf-8') as stream:
            contents = yaml.safe_load(stream)
    except FileNotFoundError:
        if explicit_path is not None:
            raise ConfigurationError('Cannot read logging configuration') from None
        contents = {}
    except (OSError, ValueError, RuntimeError, yaml.YAMLError):
        raise ConfigurationError('Cannot read logging configuration') from None
    if not isinstance(contents, dict):
        raise ConfigurationError('Logging configuration must be a mapping')
    values = contents.get(selected, {})
    if not isinstance(values, dict) or set(values) - {'directory', 'level', 'format', 'console'}:
        raise ConfigurationError('Invalid logging environment settings')
    values = dict(values)
    for key in ('directory', 'level', 'format', 'console'):
        value = env.get('RYPIQ_LOG_' + ('DIR' if key == 'directory' else key.upper()))
        if value is not None:
            if key == 'console':
                if value.lower() not in ('true', 'false', '1', '0'):
                    raise ConfigurationError('Invalid logging console setting')
                value = value.lower() in ('true', '1')
            values[key] = value
    values.update({key: value for key, value in dict(directory=directory, level=level, format=format, console=console).items() if value is not None})
    return validate_settings(LoggingSettings(environment=selected, **values), project_root=root)
