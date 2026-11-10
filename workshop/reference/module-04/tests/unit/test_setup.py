from ragworkshop.cli import main
from ragworkshop.config import Settings


def test_info(capsys):
    assert main(["info"]) == 0
    assert capsys.readouterr().out == "ragworkshop: modular ingestion and retrieval\n"


def test_default_environment():
    assert Settings().environment == "development"
