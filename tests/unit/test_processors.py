import pytest

from app.processors.base import BaseProcessor
from app.processors.text import TextProcessor

@pytest.mark.unit
def test_processor_implements_interface():
    processor = TextProcessor()

    assert isinstance(processor, BaseProcessor)
    assert callable(processor.process)

@pytest.mark.unit
def test_base_processor_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseProcessor()

@pytest.mark.unit
def test_text_processing(sample_text):
    processor = TextProcessor()
    result = processor.process(sample_text)

    assert isinstance(result, dict)
    assert result["medium"] == "text"
    assert result["content"] == sample_text
    assert result["characters"] == len(sample_text)
