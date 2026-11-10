import json
import subprocess


def test_process_and_embed_legacy_flags():
    processed = subprocess.run(["rag-workshop", "process", "ABCDEFGHIJ", "--chunk-size", "4", "--stride", "3"],
                               text=True, capture_output=True)
    assert processed.returncode == 0, processed.stderr
    assert [item["text"] for item in json.loads(processed.stdout)["chunks"]] == ["ABCD", "DEFG", "GHIJ"]
    embedded = subprocess.run(["rag-workshop", "embed", "python", "--provider", "fake", "--dimensions", "8"],
                              text=True, capture_output=True)
    assert embedded.returncode == 0, embedded.stderr
    assert json.loads(embedded.stdout)["vectors"] == [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
