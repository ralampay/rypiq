import json
import subprocess
import sys

import pytest

def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "app.cli", *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )

@pytest.mark.integration
def test_cli_process_text_file(sample_text_file):
    result = run_cli(
        "process",
        "text-file",
        sample_text_file,
        "--chunk_size=4",
        "--stride=3",
    )

    assert result.returncode == 0, result.stderr

    output = json.loads(result.stdout)

    assert output["medium"] == "text-file"
    assert output["chunk_size"] == 4
    assert output["stride"] == 3
    assert output["total_characters"] == 10
    assert output["chunk_count"] == 3

    assert [chunk["text"] for chunk in output["chunks"]] == [
        "ABCD",
        "DEFG",
        "GHIJ",
    ]

@pytest.mark.integration
@pytest.mark.parametrize(
    "options",
    [
        ("--chunk_size=0", "--stride=3"),
        ("--chunk_size=4", "--stride=0"),
        ("--chunk_size=4", "--stride=5"),
    ],
)
def test_cli_invalid_parameters(sample_text_file, options):
    result = run_cli(
        "process",
        "text-file",
        sample_text_file,
        *options,
    )

    assert result.returncode != 0

@pytest.mark.integration
@pytest.mark.parametrize(
    "options",
    [
        ("--chunk_size=4",),
        ("--stride=3",),
        (),
    ],
)
def test_cli_missing_parameterse(sample_text_file, options):
    result = run_cli(
        "process",
        "text-file",
        sample_text_file,
        *options,
    )

    assert result.returncode != 0

@pytest.mark.integration
def test_cli_missing_file(tmp_path):
    result = run_cli(
        "process",
        "text-file",
        tmp_path / "missing.txt",
        "--chunk_size=4",
        "--stride=3",
    )

    assert result.returncode != 0
    assert "error" in result.stderr.lower()

@pytest.mark.integration
def test_cli_existing_text_command():
    result = run_cli(
        "process",
        "text",
        "Hello Rypiq",
    )

    assert result.returncode == 0, result.stderr

    output = json.loads(result.stdout)

    assert output["medium"] == "text"
    assert output["words"] == 2
