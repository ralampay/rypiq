"""Worksheet rendering shared by modern and legacy Excel readers."""
from collections.abc import Iterable
from ._text import blocks, row


def worksheet(name: str, rows: Iterable[Iterable[object]]) -> str:
    content = blocks(row(values) for values in rows)
    return name + '\n' + content if content else ''

