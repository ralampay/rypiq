from dataclasses import dataclass, field
from datetime import datetime
from uuid import UUID

import pytest
from sqlalchemy import DateTime
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID as PGUUID

from app.database.schema import discover, metadata_for, persistent

def must_not_run():
    raise AssertionError("default factory evaluated during discovery")

@persistent("notes")
@dataclass
class Note:
    id: UUID
    text: str
    embedding: list[float]
    metadata: dict = field(default_factory=dict)
    label: str | None = None
    created_at: datetime = field(default_factory=must_not_run)

def test_metadata_preserves_types_and_does_not_construct_models():
    table = metadata_for([Note]).tables["notes"]
    assert isinstance(table.c.id.type, PGUUID)
    assert table.c.id.primary_key
    assert not table.c.id.nullable
    assert isinstance(table.c.embedding.type, ARRAY)
    assert isinstance(table.c.created_at.type, DateTime)
    assert table.c.created_at.type.timezone
    assert table.c.created_at.server_default is None

def test_duplicate_table_names_fail():
    @persistent("notes")
    @dataclass
    class Other:
        id: UUID

    with pytest.raises(ValueError, match="duplicate"):
        metadata_for([Note, Other])

def test_discovers_new_modules_and_excludes_unmarked_dtos(
    tmp_path, monkeypatch,
):
    package = tmp_path / "fixture_models"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "note.py").write_text(
        "from dataclasses import dataclass\n"
        "from uuid import UUID\n"
        "from app.database.schema import persistent\n"
        "@persistent('new_notes')\n"
        "@dataclass\n"
        "class NewNote:\n"
        "   id: UUID\n"
        "@dataclass\n"
        "class Wrapper:\n"
        "   text: str\n"
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    assert [m.__name__ for m in discover("fixture_models")] == [
        "NewNote"
    ]
