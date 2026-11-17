import pytest

from app.database.config import load_database_config, validate_config
from app.database.schema import build_metadata


@pytest.mark.parametrize('settings', [None, [], {'adapter': 'postgresql'},
    dict(adapter='postgresql', host='localhost', user='u', database='x', password=1),
    dict(adapter='postgresql', host='localhost', user='u', database='x', password_env='RYPIQ_NONEXISTENT_TEST_PASSWORD')])
def test_invalid_configuration_is_rejected(settings, monkeypatch):
    monkeypatch.delenv('RYPIQ_NONEXISTENT_TEST_PASSWORD', raising=False)
    with pytest.raises(ValueError):
        validate_config(settings)


@pytest.mark.parametrize('body', ['[]', 'production: {}'])
def test_configuration_shape_and_environment(tmp_path, monkeypatch, body):
    path = tmp_path / 'database.yml'
    path.write_text(body)
    monkeypatch.setenv('RYPIQ_DATABASE_CONFIG', str(path))
    monkeypatch.setenv('DATABASE_ENV', 'test')
    with pytest.raises(ValueError):
        load_database_config()


def test_schema_rejects_invalid_dimensions():
    with pytest.raises(ValueError, match='dimensions'):
        build_metadata(0)
