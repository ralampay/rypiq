from unittest.mock import Mock

import pytest

from app.operations.process import (
    PROCESSORS,
    process_content,
)

@pytest.mark.unit
def test_processor_registry():
    assert "text" in PROCESSORS

@pytest.mark.unit
def test_process_content(sample_text):
    result = process_content(
        medium="text",
        content=sample_text,
    )

    assert result["medium"] == "text"
    assert result["content"] == sample_text

@pytest.mark.unit
def test_unsupported_medium():
    with pytest.raises(
        ValueError,
        match="Unsupported medium",
    ):
        process_content(
            medium="unknown",
            content="sample",
        )

@pytest.mark.unit
def test_processor_delegation(monkeypatch):
    mock_processor = Mock()
    mock_processor.process.return_value = {
        "medium": "mock",
        "content": "test",
    }

    mock_factory = Mock(return_value=mock_processor)

    monkeypatch.setitem(
        PROCESSORS,
        "mock",
        mock_factory,
    )

    result = process_content("mock", "test")

    mock_factory.assert_called_once_with()
    mock_processor.process_assert_called_once_with("test")

    assert result["medium"] == "mock"
