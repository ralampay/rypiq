from pathlib import Path
from ragworkshop.documents.models import ExtractedDocument, TextSegment
from ragworkshop.operations import documents
from ragworkshop.processors.chunks import TextChunker


def test_chunking_resets_offsets_but_not_indexes(monkeypatch):
    class Loader:
        def load(self, path):
            return ExtractedDocument(Path("memo.txt"), [
                TextSegment("abcd", {"paragraph": 1}), TextSegment("ef", {"paragraph": 2})])
    monkeypatch.setattr(documents, "get_loader", lambda path: Loader())
    chunks = documents.process_document(Path("memo.txt"), TextChunker(3, 3))
    assert [c.index for c in chunks] == [0, 1, 2]
    assert [c.start for c in chunks] == [0, 3, 0]
    assert chunks[-1].metadata["paragraph"] == 2
