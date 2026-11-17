import json

from app import cli


def test_main_accepts_explicit_arguments(capsys):
    cli.main(['process', 'text', 'Hello world'])
    assert json.loads(capsys.readouterr().out)['words'] == 2
