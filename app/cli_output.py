"""CLI presentation and atomic JSON exports; no database or provider dependencies."""
import json
import os
from pathlib import Path
import tempfile
from app.exceptions import ValidationError
from app.resources import close_resource


def validate_export_path(path: str, *, source: str | None = None) -> Path:
    destination = Path(path).expanduser()
    if source is not None:
        original = Path(source).expanduser()
        if (destination.resolve() == original.resolve()
                or (destination.exists() and original.exists() and destination.samefile(original))):
            raise ValidationError('Export destination must differ from the input document')
    if not destination.parent.is_dir():
        raise ValidationError('Export parent directory must exist')
    if destination.exists() and not destination.is_file():
        raise ValidationError('Export destination must be a file')
    if not os.access(destination.parent, os.W_OK):
        raise ValidationError('Export parent directory must be writable')
    return destination


def write_json_export(destination: Path, value: dict) -> None:
    serialized = json.dumps(value, indent=2, allow_nan=False) + '\n'
    temporary = None
    failure = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination.parent,
                                         prefix='.rypiq-export-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
        os.replace(temporary, destination)
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if temporary is not None:
            close_resource(lambda: temporary.unlink(missing_ok=True), failure)


def embedding_preview(value: dict) -> dict:
    """Copy only presentation fields; full vectors remain intact for export."""
    if 'chunks' in value:
        return {**value, 'chunks': [embedding_preview(chunk) for chunk in value['chunks']]}
    vector = value['embedding']
    return {**value, 'embedding': vector[:3] + ['...'] + vector[-3:] if len(vector) > 6 else vector}
