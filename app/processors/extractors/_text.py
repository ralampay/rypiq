"""Text normalization shared by document readers."""
from collections.abc import Iterable
from datetime import date, datetime, time


def blocks(values: Iterable[str]) -> str:
    return '\n'.join(value for value in values if value.strip())


def cell(value: object) -> str:
    if value is None:
        return ''
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def row(values: Iterable[object]) -> str:
    cells = [cell(value) for value in values]
    while cells and not cells[-1].strip():
        cells.pop()
    return '\t'.join(cells)
