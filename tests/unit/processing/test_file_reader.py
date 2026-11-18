"""Reader selection, owned file lifetime, and sanitized errors."""
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest

from app.exceptions import DocumentExtractionError, UnsupportedFileTypeError
from app.processors.extractors.factory import EXTRACTORS, SUPPORTED_EXTENSIONS, ExtractorFactory
from app.processors.file_reader import FileTextReader


@pytest.mark.parametrize('suffix', SUPPORTED_EXTENSIONS)
def test_factory_selects_case_insensitively(suffix):
    assert ExtractorFactory().get(suffix.upper()) is EXTRACTORS[suffix]


def test_unknown_format_does_not_open_file(monkeypatch):
    opening = Mock()
    monkeypatch.setattr(Path, 'open', opening)
    with pytest.raises(UnsupportedFileTypeError) as caught:
        FileTextReader().read(Path('unknown.bin'))
    assert str(caught.value) == 'Unsupported file type; supported extensions: ' + ', '.join(SUPPORTED_EXTENSIONS)
    opening.assert_not_called()


@pytest.mark.parametrize('failure', [None, ValueError('private parser information'), OSError('read failed')])
def test_reader_uses_factory_and_closes_owned_stream(monkeypatch, failure):
    stream = BytesIO(b'document')
    monkeypatch.setattr(Path, 'open', Mock(return_value=stream))
    factory = Mock()
    extractor = factory.get.return_value
    extractor.extract.side_effect = failure
    extractor.extract.return_value = 'text'
    reader = FileTextReader(factory=factory)
    factory.get.assert_not_called()
    if isinstance(failure, ValueError):
        with pytest.raises(DocumentExtractionError) as caught:
            reader.read(Path('sample.PDF'))
        assert 'private parser information' not in str(caught.value)
    elif failure is not None:
        with pytest.raises(OSError) as caught:
            reader.read(Path('sample.PDF'))
        assert caught.value is failure
    else:
        assert reader.read(Path('sample.PDF')) == 'text'
    factory.get.assert_called_once_with('.pdf')
    extractor.extract.assert_called_once_with(stream)
    assert stream.closed


def test_open_failure_preserves_error_without_invoking_extractor(monkeypatch):
    failure = PermissionError('file access denied')
    monkeypatch.setattr(Path, 'open', Mock(side_effect=failure))
    factory = Mock()
    with pytest.raises(PermissionError) as caught:
        FileTextReader(factory=factory).read(Path('sample.pdf'))
    assert caught.value is failure
    factory.get.return_value.extract.assert_not_called()
