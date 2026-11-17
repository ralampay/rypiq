import json
import os
import subprocess
import sys

import pytest


def run_cli(*args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "app.cli", *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


@pytest.mark.integration
def test_storage_cli_save_and_list(tmp_path):
    source = tmp_path / "sample.txt"
    source.write_text("Hello storage!")

    config_file = tmp_path / "storage.yml"

    config_file.write_text(
        "test:\n"
        "  adapter: local\n"
        f"  root: {json.dumps(str(tmp_path / 'objects'))}\n"
    )

    env = {
        **os.environ,
        "STORAGE_ENV": "test",
        "RYPIQ_STORAGE_CONFIG": str(config_file),
    }

    saved = run_cli(
        "storage:save",
        str(source),
        env=env,
    )

    assert saved.returncode == 0, saved.stderr

    saved_object = json.loads(saved.stdout)

    listed = run_cli(
        "storage:list",
        env=env,
    )

    assert listed.returncode == 0, listed.stderr

    objects = json.loads(listed.stdout)

    assert len(objects) == 1
    assert objects[0]["key"] == saved_object["key"]
