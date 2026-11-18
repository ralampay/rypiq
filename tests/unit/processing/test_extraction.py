from io import BytesIO
from pathlib import Path
from unittest.mock import Mock
from types import SimpleNamespace

import pytest

from app.processors import extraction
from app.processors.extractors import spreadsheet, excel, xls
from app.operations.processing.file import process_text_file


@pytest.mark.parametrize('suffix', extraction.SUPPORTED_EXTENSIONS)
def test_missing_file_preserves_filesystem_error(tmp_path, suffix):
    with pytest.raises(FileNotFoundError):
        process_text_file(str(tmp_path / ('missing' + suffix)), 10, 10)


def test_unsupported_extension(tmp_path):
    with pytest.raises(ValueError, match='Unsupported file type'):
        process_text_file(str(tmp_path / 'unsupported.doc'), 10, 10)


@pytest.mark.parametrize('chunk_size,stride', [(0, 1), (1, 2), (True, 1), (1, False)])
def test_invalid_chunking_does_not_open_file(tmp_path, monkeypatch, chunk_size, stride):
    opening = Mock()
    monkeypatch.setattr(Path, 'open', opening)
    with pytest.raises(ValueError, match='positive integer'):
        process_text_file(str(tmp_path / 'file.pdf'), chunk_size, stride)
    opening.assert_not_called()


@pytest.mark.parametrize('fails', [False, True])
@pytest.mark.parametrize('suffix', list(extraction.EXTRACTORS))
def test_source_stream_closed(monkeypatch, suffix, fails):
    stream = BytesIO(b'contents')
    monkeypatch.setattr(Path, 'open', Mock(return_value=stream))
    parser = Mock(side_effect=ValueError('private parser details')) if fails else Mock(return_value='text')
    monkeypatch.setitem(extraction.EXTRACTORS, suffix, SimpleNamespace(extract=parser))
    if fails:
        with pytest.raises(ValueError, match='Cannot extract') as error:
            extraction.extract_file(Path('sample' + suffix))
        assert 'private parser details' not in str(error.value)
    else:
        assert extraction.extract_file(Path('sample' + suffix)) == 'text'
    parser.assert_called_once_with(stream)
    assert stream.closed


@pytest.mark.parametrize('fails', [False, True])
def test_modern_workbook_closed(monkeypatch, fails):
    book = Mock()
    sheet = Mock(title='Data')
    sheet.iter_rows.side_effect = ValueError('broken') if fails else None
    sheet.iter_rows.return_value = [('text',)]
    book.worksheets = [sheet]
    monkeypatch.setattr(excel, 'load_workbook', Mock(return_value=book))
    if fails:
        with pytest.raises(ValueError):
            spreadsheet.extract_excel(BytesIO())
    else:
        assert spreadsheet.extract_excel(BytesIO()) == 'Data\ntext'
    book.close.assert_called_once()


@pytest.mark.parametrize('fails', [False, True])
def test_legacy_workbook_released(monkeypatch, fails):
    book = Mock()
    book.sheets.side_effect = ValueError('broken') if fails else None
    book.sheets.return_value = []
    monkeypatch.setattr(xls.xlrd, 'open_workbook', Mock(return_value=book))
    if fails:
        with pytest.raises(ValueError):
            spreadsheet.extract_xls(BytesIO())
    else:
        assert spreadsheet.extract_xls(BytesIO()) == ''
    book.release_resources.assert_called_once()


@pytest.mark.parametrize('filename, content, expected', [
    ('file.unknown', b'text', 'UnsupportedFileTypeError'),
    ('file.pdf', b'not a pdf', 'DocumentExtractionError'),
    ('file.txt', b'\xff', 'DocumentExtractionError'),
])
def test_extraction_failures_have_public_types(tmp_path, filename, content, expected):
    from app.processors.extraction import extract_file
    from app.exceptions import ProcessingError, RyPiqError
    path = tmp_path / filename
    path.write_bytes(content)
    with pytest.raises(ProcessingError) as caught:
        extract_file(path)
    assert type(caught.value).__name__ == expected
    assert isinstance(caught.value, ValueError)
    assert isinstance(caught.value, RyPiqError)


@pytest.mark.parametrize('content', [b'', b'caf\xc3\xa9\r\nnext\rlast\n', b'\xef\xbb\xbftext'])
def test_text_extraction_preserves_read_text_semantics(tmp_path, content):
    path = tmp_path / 'sample.TXT'
    path.write_bytes(content)
    assert extraction.extract_file(path) == path.read_text(encoding='utf-8')


def test_invalid_text_keeps_specific_diagnostic(tmp_path):
    from app.exceptions import DocumentExtractionError
    path = tmp_path / 'sample.txt'
    path.write_bytes(b'\xff')
    with pytest.raises(DocumentExtractionError, match='text must be valid UTF-8'):
        extraction.extract_file(path)
