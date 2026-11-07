from app.processors.text_file import TextFileProcessor

def process_text_file(
    path: str,
    chunk_size: int,
    stride: int,
) -> dict:
    processor = TextFileProcessor(
        chunk_size=chunk_size,
        stride=stride,
    )

    return processor.process(path)
