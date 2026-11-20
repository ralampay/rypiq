"""Environment-specific PostgreSQL configuration; credentials stay out of URLs/logs."""
import os
from pathlib import Path

import yaml
from app.environment import load_environment

from app.exceptions import DatabaseConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validate_database_target(database, maintenance):
    for name in (database, maintenance):
        if (not isinstance(name, str) or not name.strip() or '\x00' in name
                or len(name.encode('utf-8')) > 63):
            raise DatabaseConfigurationError('Database names must contain 1 to 63 UTF-8 bytes and no NUL characters')
    if database in {maintenance, 'postgres', 'template0', 'template1'}:
        raise DatabaseConfigurationError('Target must differ from maintenance and system databases')


def validate_config(settings, *, environ=None):
    if not isinstance(settings, dict):
        raise DatabaseConfigurationError('Database environment must be a mapping')
    config = dict(settings)
    if config.get('adapter') != 'postgresql':
        raise DatabaseConfigurationError('Only the postgresql adapter is supported')
    for key in ('host', 'database', 'user'):
        if not isinstance(config.get(key), str) or not config[key].strip() or '\x00' in config[key]:
            raise DatabaseConfigurationError(f'PostgreSQL requires a valid {key}')
    maintenance = config.setdefault('maintenance_database', 'postgres')
    validate_database_target(config['database'], maintenance)
    for key, default, maximum in (('port', 5432, 65535), ('dimensions', 768, 16000)):
        value = config.setdefault(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise DatabaseConfigurationError(f'{key} must be an integer between 1 and {maximum}')
    if 'password_env' in config:
        values = os.environ if environ is None else environ
        name = config['password_env']
        if not isinstance(name, str) or name not in values:
            raise DatabaseConfigurationError('Configured password environment variable is missing' +
                (f'; export {name} before running RyPiq' if isinstance(name, str) and name.isidentifier() else ''))
        config['password'] = values[name]
    if config.get('password') is not None and not isinstance(config['password'], str):
        raise DatabaseConfigurationError('Database password must be a string')
    return config


def load_database_config(*, environ=None, project_root=None):
    if environ is None:
        load_environment()
    values = os.environ if environ is None else environ
    root = PROJECT_ROOT if project_root is None else Path(project_root)
    path = Path(values.get('RYPIQ_DATABASE_CONFIG', root / 'config/database.yml')).expanduser().resolve()
    environment = values.get('DATABASE_ENV', 'development')
    try:
        with path.open(encoding='utf-8') as stream:
            settings = yaml.safe_load(stream)
    except (OSError, UnicodeError, yaml.YAMLError):
        raise DatabaseConfigurationError('Cannot read database configuration; check path permissions and YAML syntax') from None
    if not isinstance(settings, dict):
        raise DatabaseConfigurationError('Invalid database configuration')
    if environment not in settings:
        raise DatabaseConfigurationError('Unknown database environment')
    return validate_config(settings[environment], environ=values)
