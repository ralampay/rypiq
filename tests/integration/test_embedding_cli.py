
import json
import sys

import pytest

from app import cli


@pytest.mark.integration
def test_embed_cli_dispatch(
    monkeypatch, capsys, sample_text_file
):
    seen = {}

    def fake_embed(**kwargs):
        seen.update(kwargs)

        return {
            "model": "fake.gguf",
            "chunks": [],
            "dimensions": 2,
        }

    monkeypatch.setattr(
        cli, "embed_text_file", fake_embed
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rypiq",
            "embed",
            "text-file",
            str(sample_text_file),
            "--chunk_size=4",
            "--stride=3",
            "--model=fake.gguf",
        ],
    )

    cli.main()

    assert seen["path"] == str(sample_text_file)
    assert seen["chunk_size"] == 4
    assert seen["stride"] == 3
    assert seen["model_path"] == "fake.gguf"

    output = json.loads(capsys.readouterr().out)
    assert output["model"] == "fake.gguf"


@pytest.mark.integration
def test_embed_cli_output(
    monkeypatch, capsys, sample_text_file, tmp_path
):
    monkeypatch.setattr(
        cli,
        "embed_text_file",
        lambda **kwargs: {
            "model": "fake.gguf",
            "chunks": [],
        },
    )

    output = tmp_path / "vectors.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rypiq",
            "embed",
            "text-file",
            str(sample_text_file),
            "--chunk_size=4",
            "--stride=3",
            f"--output={output}",
        ],
    )

    cli.main()

    assert json.loads(
        output.read_text(encoding="utf-8")
    )["model"] == "fake.gguf"

    assert capsys.readouterr().out == ""


@pytest.mark.integration
def test_embed_cli_missing_stride(
    monkeypatch, sample_text_file
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rypiq",
            "embed",
            "text-file",
            str(sample_text_file),
            "--chunk_size=4",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
