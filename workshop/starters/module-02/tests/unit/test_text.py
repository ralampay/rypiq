import pytest

from ragworkshop.processors.chunks import TextChunker
from ragworkshop.processors.formats import PROCESSORS, get_processor
from ragworkshop.embeddings.fake import FakeEmbeddingProvider
from ragworkshop.embeddings.factory import get_embedding_provider
from ragworkshop.embeddings.base import validate_vectors
from ragworkshop.operations.embedding import embed_chunks


def test_boundaries():
    chunks = TextChunker(4, 3).chunk("ABCDEFGHIJ", {"source": "memo"})
    assert [(c.start, c.end, c.text) for c in chunks] == [
        (0, 4, "ABCD"), (3, 7, "DEFG"), (6, 10, "GHIJ")]
    assert [c.index for c in chunks] == [0, 1, 2]
    assert all(c.metadata == {"source": "memo"} for c in chunks)


@pytest.mark.parametrize("size,stride", [(0, 1), (4, 0), (4, 5), (-1, 1), (True, 1), (4, 1.5)])
def test_bad_parameters(size, stride):
    with pytest.raises(ValueError):
        TextChunker(size, stride)


@pytest.mark.parametrize("text,expected", [("", []), ("ab", ["ab"]),
    ("aaaaaa", ["aaaa", "aaa"]), ("猫🐍éab", ["猫🐍éa", "ab"])])
def test_edges(text, expected):
    assert [c.text for c in TextChunker(4, 3).chunk(text)] == expected


def test_no_overlap():
    assert [c.text for c in TextChunker(3, 3).chunk("abcdefg")] == ["abc", "def", "g"]


@pytest.mark.parametrize("name,text,field,expected", [
    ("plain", "python", "text", "python"),
    ("markdown", "# python", "text", "# python"),
    ("json", '{"body": "python"}', "body", "python")])
def test_formats_and_pipeline(name, text, field, expected):
    processor = get_processor(name, TextChunker(30, 30), field)
    chunks = processor.process(text, {"source": "memo"})
    assert chunks[0].text == expected
    assert chunks[0].metadata["source"] == "memo"
    vectors = embed_chunks(chunks, FakeEmbeddingProvider())
    assert vectors == [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]


@pytest.mark.parametrize("text", ["{", "[]", "{}", '{"text": 4}'])
def test_bad_json(text):
    with pytest.raises(ValueError):
        get_processor("json", TextChunker()).process(text)


def test_registry_extensible(monkeypatch):
    builder = PROCESSORS["plain"]
    monkeypatch.setitem(PROCESSORS, "custom", builder)
    assert get_processor("custom", TextChunker()).process("hi")[0].text == "hi"
    with pytest.raises(ValueError):
        get_processor("unknown", TextChunker())


def test_factory_and_fake():
    provider = get_embedding_provider()
    assert provider.embed([]) == []
    assert provider.embed(["cat", "python"]) == provider.embed(["cat", "python"])
    assert provider.embed(["cat", "python"])[0] != provider.embed(["cat", "python"])[1]
    with pytest.raises(ValueError):
        get_embedding_provider("unknown")
    with pytest.raises(ValueError):
        get_embedding_provider("local")


def test_batch_order_and_error_propagation():
    class RecordingProvider:
        model_name = "recording"
        dimensions = 1
        def __init__(self):
            self.calls = []
        def embed(self, texts):
            self.calls.append(texts)
            return [[float(ord(text[0]))] for text in texts]
    provider = RecordingProvider()
    chunks = TextChunker(1, 1).chunk("abc")
    assert embed_chunks(chunks, provider, 2) == [[97.0], [98.0], [99.0]]
    assert provider.calls == [["a", "b"], ["c"]]
    def fail(texts):
        raise RuntimeError("provider unavailable")
    provider.embed = fail
    with pytest.raises(RuntimeError, match="unavailable"):
        embed_chunks(chunks, provider)
    with pytest.raises(ValueError):
        embed_chunks(chunks, provider, 0)


@pytest.mark.parametrize("vectors,count,dimensions", [([], 1, 2), ([[1]], 1, 2),
    ([[float("nan")]], 1, 1), ([[True]], 1, 1), ([[0.0]], 1, 1)])
def test_invalid_vectors(vectors, count, dimensions):
    with pytest.raises(ValueError):
        validate_vectors(vectors, count, dimensions)
