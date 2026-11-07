from app.processors.base import BaseProcessor
from app.processors.text import TextProcessor

PROCESSORS: dict[str, type[BaseProcessor]] = {
    "text": TextProcessor,
}

def process_content(medium: str, content: str) -> dict:
    processor_class = PROCESSORS.get(medium)

    if processor_class is None:
        raise ValueError(
            f"Unsupported medium: {medium}"
        )

    processor = processor_class()

    return processor.process(content)
