from unittest.mock import Mock

import pytest

from app.operations.embedding import embed_text_file

@pytest.mark.unit
def test_embed_file_preserves_chunk_offsets(sample_text_file):
    provider = Mock()
    provider.model_name = "fake.gguf"

    provider.embed.side_effect = lambda texts: [
        [float(len(text)), 0.5] for text in texts
    ]

    factory = Mock(return_value=provider)

    result = embed_text_file(
        str(sample_text_file),
        chunk_size=4,
        stride=3,
        model_path="fake.gguf",
        batch_size=2,
        provider_factory=factory,
    )

    assert [c["text"] for c in result["chunks"]] == [
        "ABCD", "DEFG", "GHIJ"
    ]

    assert [
        (c["start"], c["end"]) for c in result["chunks"]
    ] == [
        (0, 4),
        (3, 7),
        (6, 10),
    ]

    assert result["dimensions"] == 2
    assert result["chunk_count"] == 3
    assert result["embedded_chunk_count"] == 3

    # Three chunks split across two inference calls
    assert provider.embed.call_count == 2

    assert provider.embed.call_args_list[0].args[0] == [
        "title: none | text: ABCD",
        "title: none | text: DEFG",
    ]

    factory.assert_called_once_with(
        model_path="fake.gguf",
        context_size=2048,
        gpu_layers=0,
    )

@pytest.mark.unit
def test_empty_file_skips_model(empty_text_file):
    factory = Mock()

    with pytest.raises(ValueError, match="nonempty"):
        embed_text_file(
            str(empty_text_file),
            chunk_size=4,
            stride=3,
            provider_factory=factory,
        )

    factory.assert_not_called()

@pytest.mark.unit
def test_invalid_stride(sample_text_file):
    with pytest.raises(ValueError, match="stride"):
        embed_text_file(
            str(sample_text_file),
            chunk_size=4,
            stride=5,
            provider_factory=Mock(),
        )

@pytest.mark.unit
def test_invalid_batch_size(sample_text_file):
    with pytest.raises(ValueError, match="batch_size"):
        embed_text_file(
            str(sample_text_file),
            chunk_size=4,
            stride=3,
            batch_size=0,
            provider_factory=Mock(),
        )

@pytest.mark.unit
def test_vector_count_mismatch(sample_text_file):
    provider = Mock()
    provider.embed.return_value = []

    with pytest.raises(ValueError, match="count mismatch"):
        embed_text_file(
            str(sample_text_file),
            chunk_size=4,
            stride=3,
            provider_factory=lambda **kwargs: provider,
        )
