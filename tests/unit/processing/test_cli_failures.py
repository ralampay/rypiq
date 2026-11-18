import json
from unittest.mock import Mock

import pytest

from app import cli, cli_handlers


@pytest.mark.parametrize('error, expected', [
    (RuntimeError('invalid configuration'), 'invalid configuration'),
    (Exception('sensitive unexpected detail'), 'Command failed'),
])
def test_cli_failure_is_nonzero_and_safe(monkeypatch, capsys, error, expected):
    monkeypatch.setattr(cli, 'execute', Mock(side_effect=error))
    with pytest.raises(SystemExit) as caught:
        cli.main(['process', 'text', 'hello'])
    assert caught.value.code == 2
    result = capsys.readouterr()
    assert result.out == ''
    assert expected in result.err
    assert 'sensitive unexpected detail' not in result.err


def test_process_text_rejects_chunk_options(capsys):
    with pytest.raises(SystemExit):
        cli.main(['process', 'text', 'hello', '--stride', '1'])
    assert 'chunking options require text-file' in capsys.readouterr().err


def test_embeddings_create_empty_result_has_empty_chunks(monkeypatch, capsys):
    from app.models.document import Document, DocumentDetails
    monkeypatch.setattr(cli_handlers.application, 'create_document', lambda **kwargs: DocumentDetails(Document(), []))
    cli.main(['documents:create', 'unused.txt', '--application-id', '00000000-0000-0000-0000-000000000001'])
    result = json.loads(capsys.readouterr().out)
    assert result['embedding_count'] == 0 and result['embeddings'] == []
