"""Application-owned logging; initialize once at CLI or service startup."""
from contextlib import contextmanager
from collections.abc import Iterator
import logging
from logging.handlers import WatchedFileHandler
import sys
from threading import RLock
from time import perf_counter
from pathlib import Path
from typing import Any

import structlog

from app.exceptions import ConfigurationError
from app.logging.config import LoggingSettings, load_logging_settings, validate_settings

_lock = RLock()
_owned: list[logging.Handler] = []
_settings: LoggingSettings | None = None
_namespace = logging.getLogger('rypiq')
_namespace.addHandler(logging.NullHandler())
_namespace.propagate = False


def _warning() -> None:
    try:
        sys.stderr.write('RyPiq could not write a log entry.\n')
    except Exception:
        pass


class _SafeFileHandler(WatchedFileHandler):
    def handleError(self, record):
        _warning()

    def emit(self, record):
        try:
            super().emit(record)
        except Exception:
            self.handleError(record)


class _SafeStreamHandler(logging.StreamHandler):
    def handleError(self, record):
        _warning()


def _enrich(logger, method_name, event):
    if _settings is not None:
        event['environment'] = _settings.environment
    return event


_shared = [structlog.contextvars.merge_contextvars, structlog.stdlib.add_log_level,
           structlog.stdlib.add_logger_name, structlog.processors.TimeStamper(fmt='iso', utc=True), _enrich]


def get_logger(name: str = '') -> structlog.stdlib.BoundLogger:
    """Get a structured logger without loading settings or opening files."""
    name = name if name == 'rypiq' or name.startswith('rypiq.') else 'rypiq' + ('.' + name if name else '')
    return structlog.wrap_logger(logging.getLogger(name),
                                processors=[structlog.stdlib.filter_by_level, *_shared,
                                            structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
                                wrapper_class=structlog.stdlib.BoundLogger,
                                context_class=dict, cache_logger_on_first_use=False)


def configure_logging(settings: LoggingSettings | None = None, *,
                      environment: str | None = None, config_path: str | Path | None = None,
                      directory: str | Path | None = None, level: str | None = None,
                      format: str | None = None, console: bool | None = None) -> LoggingSettings:
    """Install owned handlers; explicit settings cannot be mixed with overrides."""
    global _settings
    overrides = {key: value for key, value in dict(environment=environment, config_path=config_path,
                 directory=directory, level=level, format=format, console=console).items() if value is not None}
    if settings is not None and overrides:
        raise ConfigurationError('Pass logging settings or overrides, not both')
    resolved = validate_settings(settings) if settings is not None else load_logging_settings(**overrides)
    with _lock:
        if resolved == _settings:
            return resolved
        handlers = []
        try:
            resolved.directory.mkdir(parents=True, exist_ok=True)
            handlers.append(_SafeFileHandler(resolved.directory / f'{resolved.environment}.log', encoding='utf-8'))
            if resolved.console:
                handlers.append(_SafeStreamHandler(sys.stderr))
            renderer = (structlog.processors.JSONRenderer() if resolved.format == 'json'
                        else structlog.processors.KeyValueRenderer(key_order=['timestamp', 'level', 'event']))
            formatter = structlog.stdlib.ProcessorFormatter(
                foreign_pre_chain=[structlog.stdlib.ExtraAdder(), *_shared],
                processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer])
            for handler in handlers:
                handler.setFormatter(formatter)
                handler.setLevel(resolved.level)
        except Exception:
            for handler in handlers:
                handler.close()
            raise ConfigurationError('Cannot initialize logging; check configuration and directory permissions') from None
        shutdown_logging()
        _settings = resolved
        _namespace.setLevel(resolved.level)
        for handler in handlers:
            _namespace.addHandler(handler)
        _owned.extend(handlers)
    return resolved


def shutdown_logging() -> None:
    """Flush and release only handlers created by this module."""
    global _settings
    with _lock:
        for handler in _owned:
            _namespace.removeHandler(handler)
            try:
                handler.flush()
                handler.close()
            except Exception:
                _warning()
        _owned.clear()
        _settings = None


@contextmanager
def log_context(**fields: Any) -> Iterator[None]:
    """Temporarily tag events in this execution context, restoring nested tags."""
    with structlog.contextvars.bound_contextvars(**fields):
        yield


@contextmanager
def log_operation(operation: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Record safe lifecycle events; yielded fields may hold completion metadata."""
    logger = get_logger('operations')
    started = perf_counter()
    outcome = {}
    with log_context(operation=operation, **fields):
        logger.info('operation_started')
        try:
            yield outcome
        except BaseException as exc:
            logger.error('operation_failed', error_type=type(exc).__name__,
                         elapsed_ms=round((perf_counter() - started) * 1000, 3))
            raise
        else:
            logger.info('operation_completed', **outcome,
                        elapsed_ms=round((perf_counter() - started) * 1000, 3))
