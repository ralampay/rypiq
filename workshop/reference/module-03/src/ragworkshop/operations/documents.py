from dataclasses import replace
from pathlib import Path

from ragworkshop.documents.loaders import get_loader
from ragworkshop.processors.chunks import TextChunk, TextChunker


def process_document(path: Path, chunker: TextChunker) -> list[TextChunk]:
    document = get_loader(path).load(path)
    chunks = []
    for segment_index, segment in enumerate(document.segments):
        metadata = {**segment.metadata, "source": str(document.source),
                    "filename": document.source.name, "segment_index": segment_index}
        for chunk in chunker.chunk(segment.text, metadata):
            if chunk.text.strip():
                chunks.append(replace(chunk, index=len(chunks)))
    return chunks
