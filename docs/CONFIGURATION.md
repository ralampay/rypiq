# Injectable configuration

`app.config.AppConfig` is the shared, immutable configuration snapshot for the API,
CLI, worker, and configured Python entrypoints. `load_config()` reads `.env`, the
process environment, and the existing database/storage/logging/model YAML files.
It does not create database connections, storage directories, provider clients,
or models, and does not modify `os.environ`.

The API loads once when its factory runs. The CLI loads once per invocation, and
the worker reuses one snapshot across all polling cycles. Restart the service or
worker to apply changes to environment variables or configuration files. Database
records remain live: application model and chunking updates still take effect
when the next operation reads the application.

## Load and override

```python
from dataclasses import replace

from app import create_app
from app.config import load_config

config = load_config()
config = replace(
    config,
    api=replace(config.require_api(), cors_origins=("http://localhost:5173",)),
)
api = create_app(config=config)
```

Explicit injection wins: `create_app(config=...)` and library functions supplied
with a snapshot do not reload configuration or fall back to ambient credentials.
There is no global settings singleton or implicit service locator.

For isolated loading, pass `load_config(environ={...}, project_root=...)`. The
supplied mapping replaces the process-environment input; `.env` from that project
root still supplies defaults. Construct `AppConfig` directly to avoid all file
loading.

## Construct settings directly

```python
from pathlib import Path

from app import create_app
from app.config import AppConfig, ApiSettings, DatabaseSettings, StorageSettings
from app.logging.config import LoggingSettings

config = AppConfig(
    api=ApiSettings(secret_key="replace-with-a-generated-secret"),
    database=DatabaseSettings(
        host="127.0.0.1",
        user="rypiq",
        database="rypiq_development",
        password="replace-with-your-database-password",
    ),
    storage=StorageSettings(root=Path("objects")),
    logging=LoggingSettings(directory=Path("logs")),
    catalog={},
    credentials={"MY_PROVIDER_KEY": "replace-with-your-provider-key"},
)
api = create_app(config=config)
```

Settings are frozen dataclasses. Use `dataclasses.replace` to derive a new snapshot
or replace one section. Database passwords, the service secret, and the credential
mapping are hidden from representations. Do not serialize or log the complete
configuration; secrets remain accessible to the components that need them.

The sections are:

| Section | Type / behavior |
| --- | --- |
| `api` | `ApiSettings`: secret, maximum upload bytes, CORS origins |
| `database` | `DatabaseSettings`: resolved PostgreSQL settings; 768 dimensions by default |
| `storage` | `StorageSettings`: local storage root |
| `logging` | Existing `LoggingSettings`; resolved paths and validation captured in the snapshot |
| `catalog` | Read-only model aliases using existing `CatalogEntry` values |
| `credentials` | Read-only captured environment values, resolved by `api_key_env` references |
| `issues` | Safe section-specific load/validation errors |
| `sources` | Configuration-file locations used by diagnostics |

Catalog options and credentials are defensively copied. `require_catalog()` returns
an independent dictionary of entries/options for consumers that need to merge
model settings. Configuration never owns application resources.

## Use the same snapshot throughout a workflow

```python
from uuid import UUID

from app.application import search_documents
from app.config import load_config
from app.context import create_context
from app.worker import run_once

config = load_config()
matches = search_documents(
    application_id=UUID("11111111-1111-1111-1111-111111111111"),
    query="What does the report say?",
    config=config,
)

with create_context(config=config) as context:
    documents = context.documents.list()

processed = run_once(config=config)
```

Configured document, application, embedding, search, chat, discovery, and queue
functions accept optional keyword-only `config`. Without one, each independent
library call creates its own snapshot. Pass the same instance to share a stable
configuration across several calls. Pure operations and functions that already
receive all their dependencies keep those explicit inputs.

`run_worker(config=config)` reuses the supplied snapshot until shutdown.
`cli.main(argv, config=config)` also accepts injection. The normal
`rypiq server --reload` command remains supported; programmatic CLI server
injection cannot be combined with `--reload`, because Uvicorn recreates its app
in a separate process.

Database/storage factories accept either the complete snapshot, their typed
section, or their existing dictionary arguments. Existing no-argument factory
calls remain supported. Legacy subsystem loaders also remain available; unlike
`load_config()`, their no-argument paths retain the existing `.env` preload behavior.
Provider factories accept `credentials=` for explicit lookup, including an empty
mapping to prohibit environment fallback. Existing direct provider calls without
that argument retain their legacy environment lookup.

FastAPI exposes the snapshot as `app.state.config`. Controllers obtain it through
`app.dependencies.config.get_config`, using `Depends`/`ConfigDependency`. Tests can
override that dependency for operation-specific settings. Authentication and CORS
are configured when the app is created; use a separate app instance to test other
service credentials or origins.

## Validation and diagnostics

Loading captures every section once. A missing or invalid section is recorded
without preventing unrelated capabilities from running. `require_database()`,
`require_storage()`, `require_catalog()`, `require_logging()`, and `require_api()`
raise the corresponding configuration error when that capability is needed.
Replacing a failed section with a valid explicit value makes it usable immediately.

For example, text processing does not need database settings or `APP_SECRET_KEY`;
the HTTP factory requires API settings, and starting its lifespan requires logging.
Direct model IDs/paths do not require a valid alias catalog.

`check_configuration(config=config)` reports captured configuration problems and
performs fresh database, storage-permission, and local-model-readability checks.
It does not reload YAML or `.env`, modify data, or load models. It remains usable
with incomplete configuration. Logging still has its existing process-wide
lifecycle; this change does not introduce separate loggers per API instance.

Existing variable names, YAML shapes, default values, and relative-path conventions
remain supported. Process values take precedence over `.env`; logging's explicit
arguments take precedence over its environment overrides and YAML settings.
