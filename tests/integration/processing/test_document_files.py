"""Real parser boundaries; fixtures are synthetic and contain no user data."""
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile

import pytest
from docx import Document
from openpyxl import Workbook
from pptx import Presentation
from pptx.util import Inches
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from app.cli import main
from app.application import embed_file
from app.models.application import Application
from app.operations.documents.ingest import IngestDocument
from app.operations.processing.file import process_text_file


def pdf(path, *, blank=False, encrypted=False):
    writer = PdfWriter()
    for text in ['Alpha', 'Beta']:
        page = writer.add_blank_page(width=200, height=200)
        if not blank:
            font = DictionaryObject({
                NameObject('/Type'): NameObject('/Font'),
                NameObject('/Subtype'): NameObject('/Type1'),
                NameObject('/BaseFont'): NameObject('/Helvetica'),
            })
            page[NameObject('/Resources')] = DictionaryObject({
                NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})
            })
            stream = DecodedStreamObject()
            stream.set_data(f'BT /F1 12 Tf 10 100 Td ({text}) Tj ET'.encode())
            page[NameObject('/Contents')] = writer._add_object(stream)
    if encrypted:
        writer.encrypt('secret')
    writer.write(path)


@pytest.fixture(params=['.PDF', '.docx', '.xlsx', '.xlsm', '.xls', '.pptx'])
def document_file(request, tmp_path):
    suffix = request.param
    path = tmp_path / ('sample' + suffix)
    if suffix == '.PDF':
        pdf(path)
        expected = 'Alpha\nBeta'
    elif suffix == '.docx':
        doc = Document()
        doc.add_paragraph('Alpha')
        table = doc.add_table(rows=1, cols=2)
        table.cell(0, 0).text = 'Cell'
        table.cell(0, 1).text = 'Other'
        nested = table.cell(0, 0).add_table(rows=1, cols=1)
        nested.cell(0, 0).text = 'Nested'
        doc.add_paragraph('Beta')
        doc.sections[0].header.paragraphs[0].text = 'Excluded'
        doc.save(path)
        expected = 'Alpha\nCell\nNested\tOther\nBeta'
    elif suffix in ('.xlsx', '.xlsm'):
        book = Workbook()
        sheet = book.active
        sheet.title = 'Data'
        sheet.append(['Alpha', None, 42, True])
        sheet.append([datetime(2024, 1, 2)])
        sheet.append(['=1+1'])
        book.create_sheet('Empty')
        hidden = book.create_sheet('Hidden')
        hidden.sheet_state = 'hidden'
        hidden.append(['Beta'])
        book.save(path)
        book.close()
        # Make an actual macro-enabled package, rather than only rename XLSX.
        if suffix == '.xlsm':
            with ZipFile(path) as archive:
                entries = {name: archive.read(name) for name in archive.namelist()}
            entries['[Content_Types].xml'] = entries['[Content_Types].xml'].replace(
                b'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml',
                b'application/vnd.ms-excel.sheet.macroEnabled.main+xml')
            with ZipFile(path, 'w') as archive:
                for name, data in entries.items():
                    archive.writestr(name, data)
        expected = 'Data\nAlpha\t\t42\ttrue\n2024-01-02T00:00:00\nHidden\nBeta'
    elif suffix == '.xls':
        path.write_bytes((Path(__file__).parent / 'fixtures/sample.xls').read_bytes())
        expected = 'Data\nAlpha\t\t42\ttrue\n2024-01-02T00:00:00\nHidden\nBeta'
    else:
        deck = Presentation()
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        group = slide.shapes.add_group_shape()
        group.shapes.add_textbox(0, 0, Inches(2), Inches(1)).text = 'Alpha'
        table = slide.shapes.add_table(1, 2, 0, Inches(1), Inches(4), Inches(1)).table
        table.cell(0, 0).text = 'Cell'
        table.cell(0, 1).text = 'Other'
        slide.notes_slide.notes_text_frame.text = 'Excluded'
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        slide.shapes.add_textbox(0, 0, Inches(2), Inches(1)).text = 'Beta'
        deck.save(path)
        expected = 'Alpha\nCell\tOther\nBeta'
    return path, expected


def test_real_extraction_and_offsets(document_file):
    path, expected = document_file
    result = process_text_file(str(path), 9, 7)
    assert result['medium'] == 'text-file'
    assert result['total_characters'] == len(expected)
    assert result['chunks']
    for chunk in result['chunks']:
        assert chunk['text'] == expected[chunk['start']:chunk['end']]
    whole = process_text_file(str(path), 1000, 1000)
    assert whole['chunks'][0]['text'] == expected


def test_cli_process(document_file, capsys):
    path, expected = document_file
    main(['process', 'text-file', str(path), '--chunk_size', '1000', '--stride', '1000'])
    assert json.loads(capsys.readouterr().out)['chunks'][0]['text'] == expected


@pytest.fixture
def document_resources(tmp_path):
    from contextlib import nullcontext
    from dataclasses import replace
    from app.models.document import DocumentDetails
    from app.storage.providers.local import LocalStorage
    repository = Mock(name_exists=Mock(return_value=False))
    records = {}
    def save(document):
        records[document.id] = replace(document)
        return replace(document)
    def status(identifier, value):
        return save(replace(records[identifier], status=value))
    repository.create.side_effect = repository.update.side_effect = save
    repository.set_status.side_effect = status
    repository.processing_lock.side_effect = lambda identifier: nullcontext()
    repository.replace_embeddings.side_effect = lambda document, chunks, **kwargs: DocumentDetails(status(document.id, 'done'), chunks)
    return repository, Mock(wraps=LocalStorage(tmp_path / 'objects'))


def test_embedding_and_ingestion(document_file, document_resources):
    path, expected = document_file
    original = path.read_bytes()
    provider = Mock(model_name='fake.gguf')
    provider.embed.side_effect = lambda texts: [[1., 0., 0.] for _ in texts]
    result = embed_file(path=str(path), chunk_size=1000, stride=1000,
                        provider_factory=Mock(return_value=provider))
    assert result['chunks'][0]['text'] == expected
    provider.embed.assert_called_with(['title: none | text: ' + expected])
    provider.reset_mock()
    repository, storage = document_resources
    docs = IngestDocument(path=str(path), chunk_size=1000, stride=1000,
                          application=Application(name='test', embedding_model='fake.gguf', embedding_size=3),
                          embedding_provider=provider, storage=storage, repository=repository, name='test document').execute()
    assert docs.embeddings[0].content == expected
    assert docs.document.filename == path.name
    assert storage.save.call_args.kwargs['source'] == path
    assert path.read_bytes() == original
    provider.close.assert_not_called()


@pytest.mark.parametrize('suffix', ['pdf', 'docx', 'xlsx', 'xlsm', 'xls', 'pptx'])
def test_corrupt_file_retains_original_without_embeddings(tmp_path, suffix, document_resources):
    path = tmp_path / ('bad.' + suffix)
    path.write_bytes(b'not a document')
    factory, provider = Mock(), Mock(model_name='fake.gguf')
    repository, storage = document_resources
    with pytest.raises(ValueError, match='Cannot extract'):
        embed_file(path=str(path), chunk_size=10, stride=10, provider_factory=factory)
    factory.assert_not_called()
    with pytest.raises(ValueError, match='Cannot extract'):
        IngestDocument(path=str(path), chunk_size=10, stride=10,
                       application=Application(name='test', embedding_model='fake.gguf', embedding_size=3),
                       embedding_provider=provider, storage=storage, repository=repository, name='test document').execute()
    provider.embed.assert_not_called()
    storage.save.assert_called_once()
    repository.replace_embeddings.assert_not_called()
    assert repository.set_status.call_args.args[1] == 'error'
    assert len(storage.list()) == 1


@pytest.mark.parametrize('encrypted', [False, True])
def test_pdf_without_accessible_text(tmp_path, encrypted):
    path = tmp_path / 'empty.pdf'
    pdf(path, blank=True, encrypted=encrypted)
    factory = Mock()
    with pytest.raises(ValueError, match='encrypted' if encrypted else 'nonempty'):
        embed_file(path=str(path), chunk_size=10, stride=10, provider_factory=factory)
    factory.assert_not_called()


def test_cached_formula_values(tmp_path):
    path = tmp_path / 'formulas.xlsx'
    book = Workbook()
    book.active.append(['=1+1', '=2+2'])
    book.save(path)
    book.close()
    with ZipFile(path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist()}
    entries['xl/worksheets/sheet1.xml'] = entries['xl/worksheets/sheet1.xml'].replace(
        b'<f>1+1</f><v></v>', b'<f>1+1</f><v>2</v>')
    with ZipFile(path, 'w') as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    result = process_text_file(str(path), 1000, 1000)
    assert result['chunks'][0]['text'] == 'Sheet\n2'


@pytest.mark.parametrize('suffix', ['docx', 'xlsx', 'pptx'])
def test_empty_office_files(tmp_path, suffix):
    path = tmp_path / ('empty.' + suffix)
    {'docx': Document, 'xlsx': Workbook, 'pptx': Presentation}[suffix]().save(path)
    assert process_text_file(str(path), 10, 10)['chunks'] == []
    factory = Mock()
    with pytest.raises(ValueError, match='nonempty'):
        embed_file(path=str(path), chunk_size=10, stride=10, provider_factory=factory)
    factory.assert_not_called()


def test_cli_corrupt_document_is_safe(tmp_path, capsys):
    path = tmp_path / 'bad.docx'
    path.write_bytes(b'private contents')
    with pytest.raises(SystemExit) as error:
        main(['process', 'text-file', str(path), '--chunk_size', '10', '--stride', '10'])
    assert error.value.code != 0
    output = capsys.readouterr()
    assert not output.out
    assert 'Cannot extract .docx' in output.err
    assert 'private contents' not in output.err


def test_readers_borrow_streams_and_keep_function_entrypoints(document_file):
    from app.processors.extraction import EXTRACTORS
    from app.processors.extractors import pdf, word, spreadsheet, presentation

    path, expected = document_file
    suffix = path.suffix.lower()
    compatibility = {
        '.pdf': pdf.extract, '.docx': word.extract,
        '.xlsx': spreadsheet.extract_excel, '.xlsm': spreadsheet.extract_excel,
        '.xls': spreadsheet.extract_xls, '.pptx': presentation.extract,
    }
    with path.open('rb') as stream:
        assert EXTRACTORS[suffix].extract(stream) == expected
        assert not stream.closed
        stream.seek(0)
        assert compatibility[suffix](stream) == expected
        assert not stream.closed
