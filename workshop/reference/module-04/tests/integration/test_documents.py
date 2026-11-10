from pathlib import Path
import subprocess
import json

import pytest
from docx import Document
from reportlab.pdfgen.canvas import Canvas

from ragworkshop.documents.loaders import get_loader
from ragworkshop.operations.documents import process_document
from ragworkshop.processors.chunks import TextChunker


@pytest.fixture
def documents(tmp_path):
    txt = tmp_path / "lesson.txt"
    txt.write_text("python modules", encoding="utf-8")
    docx = tmp_path / "lesson.docx"
    document = Document()
    document.add_paragraph("python modules")
    document.add_paragraph("database storage")
    document.save(docx)
    pdf = tmp_path / "lesson.pdf"
    canvas = Canvas(str(pdf), invariant=1)
    canvas.drawString(72, 720, "python modules")
    canvas.showPage()
    canvas.drawString(72, 720, "database storage")
    canvas.save()
    return [txt, docx, pdf]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_real_parsers(documents, index):
    path = documents[index]
    chunks = process_document(path, TextChunker(100, 100))
    assert "python modules" in chunks[0].text
    assert chunks[0].metadata["source"] == str(path)
    assert chunks[0].start == 0
    if index == 1:
        assert chunks[1].metadata["paragraph"] == 2
    if index == 2:
        assert chunks[1].metadata["page"] == 2
    result = subprocess.run(["rag-workshop", "process-file", str(path)],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "python modules" in json.loads(result.stdout)["chunks"][0]["text"]


def test_missing_unsupported_and_empty(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_loader(tmp_path / "absent.txt").load(tmp_path / "absent.txt")
    with pytest.raises(ValueError, match="Unsupported"):
        get_loader(tmp_path / "file.exe")
    path = tmp_path / "empty.txt"
    path.write_text(" ", encoding="utf-8")
    assert process_document(path, TextChunker()) == []


@pytest.mark.parametrize("suffix", [".pdf", ".docx"])
def test_corrupt_file(tmp_path, suffix):
    path = tmp_path / ("bad" + suffix)
    path.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="Cannot parse"):
        get_loader(path).load(path)


def test_unreadable(monkeypatch, tmp_path):
    path = tmp_path / "locked.txt"
    path.touch()
    def denied(self, **kwargs):
        raise PermissionError("denied")
    monkeypatch.setattr(Path, "read_text", denied)
    with pytest.raises(PermissionError):
        get_loader(path).load(path)
