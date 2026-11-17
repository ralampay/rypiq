from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[3]


def test_project_dependencies_match_cli_postgresql_application():
    project = tomllib.loads((ROOT / 'pyproject.toml').read_text())['project']
    dependencies = project['dependencies']
    for removed in ('fastapi', 'uvicorn', 'chromadb', 'alembic'):
        assert not any(removed in requirement.lower() for requirement in dependencies)
    for required in ('sqlalchemy', 'pgvector', 'psycopg'):
        assert any(required in requirement.lower() for requirement in dependencies)


def test_obsolete_scaffolding_is_removed():
    for relative in ('workshop', 'alembic.ini', 'db/postgresql', 'app/routes.py',
                     'app/database/alembic_generation.py', 'app/database/chroma_migrations.py',
                     'app/database/backends/chroma.py', 'app/database/repositories/chroma.py'):
        assert not (ROOT / relative).exists(), relative
