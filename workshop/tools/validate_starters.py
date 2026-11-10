"""Prove every exercise overlay produces behavioral Red on a prior Green stage."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    results = []
    with tempfile.TemporaryDirectory(prefix="ragworkshop-red-") as folder:
        for number in range(1, 6):
            project = Path(folder) / f"module-{number:02d}"
            project.mkdir()
            if number > 1:
                shutil.copytree(ROOT / "reference" / f"module-{number-1:02d}", project, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.egg-info"))
            shutil.copytree(ROOT / "starters" / f"module-{number:02d}", project, dirs_exist_ok=True)
            if number == 1:
                # Packaging is completed before the intended behavioral Red checkpoint.
                shutil.copyfile(ROOT / "reference/module-01/pyproject.toml", project / "pyproject.toml")
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", "."],
                           cwd=project, check=True, stdout=subprocess.DEVNULL)
            result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=project,
                                    text=True, capture_output=True)
            assert result.returncode == 1, result.stdout + result.stderr
            assert "NotImplementedError" in result.stdout, result.stdout
            assert "ERROR collecting" not in result.stdout, result.stdout
            summary = result.stdout.splitlines()[-1]
            print(f"Module {number} expected Red: {summary}", flush=True)
            results.append({"module": number, "summary": summary})
    # Restore a usable installation before deleting the temporary exercise directories.
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-e",
                    str(ROOT / "reference/module-05")], check=True, stdout=subprocess.DEVNULL)
    Path("/tmp/ragworkshop-red-results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
