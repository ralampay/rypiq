# RyPiq

A Python CLI and library for processing text, generating local embeddings, storing original files, and searching documents with PostgreSQL and pgvector.

## Setup

Use Python 3.14.3 or newer. Persistence and search require PostgreSQL with pgvector; see [database prerequisites](docs/DATABASE.md#setup-and-configuration).

```sh
python -m venv env
. env/bin/activate
python -m pip install -e '.[dev]'
```

Configure PostgreSQL and initialize the database using the [database setup guide](docs/DATABASE.md#setup-and-configuration).

## Database

See [Database documentation](docs/DATABASE.md) for configuration, lifecycle commands, schema and migration behavior, application management, CRUD services, and database integration tests.

## Ingest and search

Provide a compatible local GGUF embedding model; these commands do not download one.

```sh
rypiq application:create --name 'My application' \
  --embedding-model models/embeddinggemma-300m-Q4_0.gguf --embedding-size 768
rypiq application:list
# Use the UUID returned by application:create:
rypiq ingest samples/sample.txt --application-id APPLICATION_UUID --chunk_size 500 --stride 400 \
  --model models/embeddinggemma-300m-Q4_0.gguf
rypiq search 'What is this document about?' --application-id APPLICATION_UUID \
  --model models/embeddinggemma-300m-Q4_0.gguf
```

See [application commands](docs/DATABASE.md#application-commands) for configuration options and supported providers.

Ingestion and search use the application's model path. Optional `--model` supplies another local path with the same filename, useful for migrated records that contain only a filename. Standalone `embed` remains independent of applications.

`config/storage.yml` controls original-file storage. `STORAGE_ENV` defaults to `DATABASE_ENV`; `RYPIQ_STORAGE_CONFIG` overrides its configuration path. Storage paths remain independent of database reset.

Use `rypiq --help` for text processing, embedding, and object-storage commands. Library callers should use `with create_context() as context:` from `app.context` or `with get_database() as database:` from `app.database.factory` to release connection pools reliably.

## Tests

Tests are grouped by domain inside `tests/unit/` and `tests/integration/`: database, embeddings, processing, storage, and operations. See [database integration tests](docs/DATABASE.md#database-integration-tests) for disposable PostgreSQL setup.

```sh
env/bin/python -m pytest tests/unit -q
env/bin/python -m pytest tests -q
```

Without `RYPIQ_TEST_ADMIN_URL`, database integrations are skipped. Model integration tests require the local model above or `EMBEDDING_MODEL_PATH`; absent models are skipped, never downloaded. A skipped integration is not evidence of verified support. See [validation results](docs/VALIDATION.md) for the recorded TDD iterations and actual results.

## Contributing and coding agents

Read [AGENTS.md](AGENTS.md) for agent instructions and the [code conventions](docs/CODE_CONVENTIONS.md) for command objects, operation functions, factories, repositories, resource ownership, and testing guidance.

See [CLI and backend execution paths](docs/EXECUTION_PATHS.md) for configured Python calls, injected commands, resource ownership, and the branch-coverage command.
