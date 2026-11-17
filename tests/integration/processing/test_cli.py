import json
import subprocess
import sys

import pytest

def run_cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "app.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )

@pytest.mark.integration
def test_cli_help():
    result = run_cli("--help")

    assert result.returncode == 0
    assert "process" in result.stdout

@pytest.mark.integration
def test_cli_process_text(sample_text):
    result = run_cli(
        "process",
        "text",
        sample_text,
    )

    assert result.returncode == 0, result.stderr

    output = json.loads(result.stdout)

    assert output["medium"] == "text"
    assert output["content"] == sample_text
    assert output["words"] == len(sample_text.split())

@pytest.mark.integration
def test_cli_unsupported_medium():
    result = run_cli(
        "process",
        "unknown",
        "sample",
    )

    assert result.returncode != 0
