"""Temporary upload ownership at the HTTP composition boundary."""
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import HTTPException, UploadFile


@contextmanager
def staged_upload(upload: UploadFile, *, limit: int):
    filename = Path((upload.filename or 'upload').replace('\\', '/')).name
    if filename in ('', '.', '..') or '\x00' in filename:
        raise HTTPException(422, 'Invalid upload filename')
    with TemporaryDirectory(prefix='rypiq-upload-') as directory:
        path = Path(directory) / filename
        size = 0
        try:
            with path.open('wb') as target:
                while chunk := upload.file.read(1024 * 1024):
                    size += len(chunk)
                    if size > limit:
                        raise HTTPException(413, 'File exceeds upload limit')
                    target.write(chunk)
            yield path
        finally:
            upload.file.close()
