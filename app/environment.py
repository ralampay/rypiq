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
