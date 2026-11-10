"""Install and test each checkpoint in the invoking Python environment.

Run with a dedicated virtual environment after installing module-05[dev].
The environment's bin directory must be on PATH for console-script tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("/tmp/ragworkshop-results.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    results = []
    for number in range(1, 6):
        project = root / "reference" / f"module-{number:02d}"
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-e", "."],
                       cwd=project, check=True, stdout=subprocess.DEVNULL)
        output = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=project,
                                text=True, capture_output=True)
        print(f"Module {number}: {output.stdout.splitlines()[-1]}", flush=True)
        results.append({"module": number, "returncode": output.returncode,
                        "summary": output.stdout.splitlines()[-1]})
        if output.returncode:
            print(output.stdout)
            print(output.stderr)
            raise SystemExit(output.returncode)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
