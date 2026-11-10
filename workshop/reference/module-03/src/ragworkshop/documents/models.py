from dataclasses import dataclass, field
from pathlib import Path
from ragworkshop.processors.chunks import Metadata


@dataclass(frozen=True)
class TextSegment:
    text: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True)
class ExtractedDocument:
    source: Path
    segments: list[TextSegment]
