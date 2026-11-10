import json
import subprocess
from dataclasses import replace

from docx import Document as DocxDocument
from reportlab.pdfgen.canvas import Canvas
import pytest

from ragworkshop.config import load_settings
from ragworkshop.context import create_context
from ragworkshop.embeddings.fake import FakeEmbeddingProvider
from ragworkshop.operations.ingestion import ingest_document
from ragworkshop.operations.search import semantic_search
from ragworkshop.processors.chunks import TextChunker


@pytest.mark.parametrize("suffix", [".txt", ".docx", ".pdf"])
def test_deterministic_fake_provider_e2e(tmp_path, suffix):
    path = tmp_path / ("lesson" + suffix)
    if suffix == ".txt":
        path.write_text("python python modules", encoding="utf-8")
    elif suffix == ".docx":
        document = DocxDocument()
        document.add_paragraph("python python modules")
        document.save(path)
    else:
        canvas = Canvas(str(path), invariant=1)
        canvas.drawString(72, 720, "python python modules")
        canvas.save()
    settings = load_settings()
    assert settings.environment == "test"
    assert settings.database_path == tmp_path / "db"
    with create_context(settings, FakeEmbeddingProvider()) as context:
        indexed = ingest_document(path, context, TextChunker(100, 100))
        assert context.storage.exists(indexed[0].storage_key)
        destination = context.storage.get(indexed[0].storage_key, tmp_path / ("copy" + suffix))
        assert destination.read_bytes() == path.read_bytes()
    with create_context(settings, FakeEmbeddingProvider()) as fresh:
        results = semantic_search("python python", fresh)
        assert results[0].document.id == indexed[0].id
        assert results[0].document.metadata["filename"] == path.name
    other = replace(settings, environment="development", database_path=tmp_path / "isolated-dev-db",
                    storage_root=tmp_path / "isolated-dev-objects")
    with create_context(other, FakeEmbeddingProvider()) as isolated:
        assert semantic_search("python", isolated) == []
        assert isolated.storage.list() == []


def test_installed_cli_persists_across_processes(tmp_path):
    source = tmp_path / "lesson.txt"
    source.write_text("python modules", encoding="utf-8")
    def run(*args):
        result = subprocess.run(["rag-workshop", *map(str, args)], text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout)
    assert run("db:create") == {"database": "initialized"}
    assert run("db:migrate") == {"applied": ["0001"]}
    assert run("db:migrate") == {"applied": []}
    indexed = run("ingest", source)
    result = run("search", "python", "--source-id", indexed["source_id"])
    assert result[0]["document"]["storage_key"] == indexed["storage_key"]
    assert run("storage:list")[0]["key"] == indexed["storage_key"]
