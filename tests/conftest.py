import pytest


@pytest.fixture(autouse=True)
def isolate_project_environment(tmp_path, monkeypatch):
    """Never preload developer credentials into tests."""
    from app import environment

    monkeypatch.setattr(environment, 'PROJECT_ROOT', tmp_path)

@pytest.fixture
def sample_text():
    return "Hello world!"

@pytest.fixture
def empty_text():
    return ""

@pytest.fixture
def sample_text_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("ABCDEFGHIJ", encoding="utf-8")

    return file_path

@pytest.fixture
def empty_text_file(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    return file_path
