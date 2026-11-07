import pytest

@pytest.fixture
def sample_text():
    return "Hello world!"

@pytest.fixture
def empty_text():
    return ""
