"""Sanitize filesystem failures at object-storage boundaries."""
from contextlib import contextmanager
from collections.abc import Iterator

from app.exceptions import (
    RyPiqError, StorageError, StorageNotFoundError,
    StorageAlreadyExistsError, StoragePermissionError,
)


@contextmanager
def storage_errors() -> Iterator[None]:
    try:
        yield
    except RyPiqError:
        raise
    except OSError as failure:
        if isinstance(failure, FileNotFoundError):
            error = StorageNotFoundError('Storage file or directory does not exist')
        elif isinstance(failure, FileExistsError):
            error = StorageAlreadyExistsError('Storage destination already exists')
        elif isinstance(failure, PermissionError):
            error = StoragePermissionError('Storage access denied; check filesystem permissions')
        else:
            error = StorageError('Storage operation failed; check filesystem availability and permissions')
        raise error from None
