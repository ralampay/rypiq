"""Run the final checkpoint with Python socket/DNS audit events blocked.

Install the checkpoint (editable or wheel) in the active environment first.
This validates Python networking; it is not an OS sandbox for native code.
"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
environment = {**os.environ, "PYTHONPATH": str(ROOT / "tools/offline_guard")}
probe = subprocess.run([sys.executable, "-c", "import socket; socket.getaddrinfo('example.com', 443)"],
                       env=environment, text=True, capture_output=True)
assert probe.returncode != 0 and "Network access blocked" in probe.stderr
result = subprocess.run([sys.executable, "-m", "pytest", "-q"],
                        cwd=ROOT / "reference/module-05", env=environment)
raise SystemExit(result.returncode)
