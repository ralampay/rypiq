"""Typed results shared by configuration and database diagnostics."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ConfigurationCheck:
    name: str
    status: Literal['COMPLETE', 'REQUIRED', 'INVALID', 'OPTIONAL']
    detail: str


@dataclass(frozen=True)
class ConfigurationReport:
    environment: str
    checks: tuple[ConfigurationCheck, ...]

    @property
    def incomplete_count(self) -> int:
        return sum(check.status in ('REQUIRED', 'INVALID') for check in self.checks)


