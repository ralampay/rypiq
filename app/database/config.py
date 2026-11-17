"""Environment-specific PostgreSQL configuration; credentials stay out of URLs/logs."""
import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def validate_database_target(database, maintenance):
    for name in (database, maintenance):
        if (not isinstance(name, str) or not name.strip() or '\x00' in name
                or len(name.encode('utf-8')) > 63):
            raise ValueError('Database names must contain 1 to 63 UTF-8 bytes and no NUL characters')
    if database in {maintenance, 'postgres', 'template0', 'template1'}:
        raise ValueError('Target must differ from maintenance and system databases')


def validate_config(settings):
    if not isinstance(settings, dict):
        raise ValueError('Database environment must be a mapping')
    config = dict(settings)
    if config.get('adapter') != 'postgresql':
        raise ValueError('Only the postgresql adapter is supported')
    for key in ('host', 'database', 'user'):
        if not isinstance(config.get(key), str) or not config[key].strip() or '\x00' in config[key]:
            raise ValueError(f'PostgreSQL requires a valid {key}')
    maintenance = config.setdefault('maintenance_database', 'postgres')
    validate_database_target(config['database'], maintenance)
    for key, default, maximum in (('port', 5432, 65535), ('dimensions', 768, 16000)):
        value = config.setdefault(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f'{key} must be an integer between 1 and {maximum}')
    if 'password_env' in config:
        name = config['password_env']
        if not isinstance(name, str) or name not in os.environ:
            raise ValueError('Configured password environment variable is missing')
        config['password'] = os.environ[name]
    if config.get('password') is not None and not isinstance(config['password'], str):
        raise ValueError('Database password must be a string')
    return config


def load_database_config():
    path = Path(os.getenv('RYPIQ_DATABASE_CONFIG', PROJECT_ROOT / 'config/database.yml')).expanduser().resolve()
    environment = os.getenv('DATABASE_ENV', 'development')
    try:
        with path.open(encoding='utf-8') as stream:
            settings = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError):
        raise ValueError('Cannot read database configuration; check the path and YAML syntax') from None
    if not isinstance(settings, dict):
        raise ValueError('Invalid database configuration')
    if environment not in settings:
        raise ValueError('Unknown database environment')
    return validate_config(settings[environment])
