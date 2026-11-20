"""Load local environment settings at configuration boundaries."""
from pathlib import Path

from dotenv import load_dotenv

from app.exceptions import ConfigurationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_environment() -> None:
    """Load project-local defaults without replacing existing environment values."""
    try:
        stream = (PROJECT_ROOT / '.env').open(encoding='utf-8')
    except FileNotFoundError:
        return
    except OSError:
        raise ConfigurationError('Cannot read .env; check file permissions and encoding') from None
    try:
        with stream:
            load_dotenv(stream=stream, override=False)
    except (OSError, UnicodeError):
        raise ConfigurationError('Cannot read .env; check file permissions and encoding') from None


def read_environment(*, environ=None, project_root: Path | None = None) -> dict[str, str]:
    """Snapshot .env defaults and explicit/process values without changing os.environ."""
    import os
    from dotenv import dotenv_values
    from dotenv.variables import parse_variables
    values = dict(os.environ if environ is None else environ)
    root = PROJECT_ROOT if project_root is None else Path(project_root)
    try:
        with (root / '.env').open(encoding='utf-8') as stream:
            defaults = dotenv_values(stream=stream, interpolate=False)
    except FileNotFoundError:
        defaults = {}
    except (OSError, UnicodeError):
        raise ConfigurationError('Cannot read .env; check file permissions and encoding') from None
    resolved = {}
    for key, value in defaults.items():
        if value is not None:
            scope = {**resolved, **values}
            resolved[key] = ''.join(atom.resolve(scope) for atom in parse_variables(value))
    return {**resolved, **values}
