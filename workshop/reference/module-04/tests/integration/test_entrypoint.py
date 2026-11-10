import subprocess


def test_installed_entrypoint(tmp_path):
    result = subprocess.run(["rag-workshop", "info"], cwd=tmp_path,
                            text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ragworkshop: modular ingestion and retrieval\n"
