import json
from uuid import UUID

import pytest

from app import cli, application
from app.models.document import Document


@pytest.mark.integration
def test_embeddings_cli_dispatch(monkeypatch, capsys, sample_text_file):
    seen = {}

    def fake_create(**kwargs):
        seen.update(kwargs)
        return [Document(content='text', embedding=[1., 0.])]

    monkeypatch.setattr(application, 'create_embeddings', fake_create)
    cli.main(['embeddings:create', str(sample_text_file), '--application-id', str(UUID(int=1))])
    assert seen == {'path': str(sample_text_file), 'application_id': UUID(int=1), 'batch_size': 1}
    output = json.loads(capsys.readouterr().out)
    assert output['chunks'][0]['embedding'] == [1., 0.]


@pytest.mark.integration
def test_embeddings_cli_output(monkeypatch, capsys, sample_text_file, tmp_path):
    monkeypatch.setattr(application, 'create_embeddings',
                        lambda **kwargs: [Document(content='text', embedding=[1., 0.])])
    output = tmp_path / 'vectors.json'
    cli.main(['embeddings:create', str(sample_text_file), '--application-id', str(UUID(int=1)),
              '--output', str(output)])
    assert json.loads(output.read_text()) == json.loads(capsys.readouterr().out)


@pytest.mark.integration
def test_embeddings_cli_requires_application(sample_text_file):
    with pytest.raises(SystemExit) as exc:
        cli.main(['embeddings:create', str(sample_text_file)])
    assert exc.value.code == 2
