# RyPiq

A Python CLI and library for processing text, generating local embeddings, storing original files, and searching documents with PostgreSQL and pgvector.

## Setup

Use Python 3.14.3 or newer. Install PostgreSQL and the pgvector server extension on the database host. The Python `pgvector` package does not install that server extension.

```sh
python -m venv env
. env/bin/activate
python -m pip install -e '.[dev]'
```

Edit `config/database.yml` for your host, port, database names, and PostgreSQL user. Export `RYPIQ_DATABASE_PASSWORD` in your shell. The default configuration uses PostgreSQL's `postgres` database for maintenance connections. The maintenance and target databases must differ.

```sh
export DATABASE_ENV=development
export RYPIQ_DATABASE_PASSWORD='your-local-password'
rypiq db:create
```

The configured role needs permission to create/drop its databases and initialize the `vector` extension. Provision the extension with an administrator if the application role cannot create it. Errors after database creation leave the database in place: correct the permissions or extension installation and retry `db:migrate`.

`DATABASE_ENV` selects `development`, `test`, or `production`. `RYPIQ_DATABASE_CONFIG` selects an alternative YAML file by absolute path, including when running outside the checkout or using a non-editable package installation. `.env.example` lists supported variables; the CLI does not automatically load `.env` files. Do not commit real credentials.

## Database commands

| Command | Behavior |
| --- | --- |
| `db:create` | Create the database if missing, enable pgvector, and create the fixed application tables. Repeating it preserves data. |
| `db:migrate` | Initialize the extension and missing tables in an existing database. Validate existing managed tables without changing their structure. |
| `db:drop` | Delete the configured target database and its data. A missing target is a no-op. |
| `db:reset` | Drop the configured target database, then create it again with the extension and application tables. |

Commands return JSON. Failures return a nonzero exit code with a concise error. Database creation/drop use an autocommit maintenance connection; extension and table initialization run in one transaction. A failed reset after the drop cannot restore the old data. Active external connections make drop/reset fail; the commands do not forcibly terminate other sessions.

**Drop and reset delete database contents, including unrelated tables in that database.** They preserve original files in object storage. Close other applications' connections before resetting, and use a dedicated application database. Run schema initialization with one writer at a time.

## Fixed schema and embeddings

The application explicitly declares `public.documents` in `app/database/schema.py`. It stores all fields of the `Document` dataclass: content, embedding, UUID identifiers, source/object references, chunk index, `model_name`, JSONB metadata, and timezone-aware timestamps. UUID/time defaults are supplied by Python when constructing a Document.

`dimensions` in `database.yml` sets the vector column size; it defaults to **768**, matching the current embedding model. Writes and searches must use finite, nonzero float32 vectors of exactly that size. Changing dimensions does not alter an existing table: initialization rejects the mismatch. For disposable data, reset the database; otherwise make an explicit, reviewed schema and data conversion.

Add future tables explicitly to the schema definition. `db:migrate` creates missing declared tables and checks existing columns, types, nullability, and primary keys. It does not generate migration files, apply column changes, or maintain a revision journal. Existing legacy storage/schema data is not automatically imported.

CRUD uses parameterized SQLAlchemy Core queries. Search orders by pgvector cosine distance in SQL, with UUID tie-breaking and scalar equality filters. `0` is an identical direction, `1` orthogonal, and `2` opposite. Search is exact; no approximate vector index is created. Large corpora may require a separate indexing/performance change. Keep ingestion and query models consistent: equal vector dimensions alone do not make different models compatible. Library callers can filter by `model_name`.

## Ingest and search

Provide a compatible local GGUF embedding model; these commands do not download one.

```sh
rypiq ingest samples/sample.txt --chunk_size 500 --stride 400 \
  --model models/embeddinggemma-300m-Q4_0.gguf
rypiq search 'What is this document about?' \
  --model models/embeddinggemma-300m-Q4_0.gguf
```

`config/storage.yml` controls original-file storage. `STORAGE_ENV` defaults to `DATABASE_ENV`; `RYPIQ_STORAGE_CONFIG` overrides its configuration path. Storage paths remain independent of database reset.

Use `rypiq --help` for text processing, embedding, and object-storage commands. Library callers should use `with create_context() as context:` from `app.context` or `with get_database() as database:` from `app.database.factory` to release connection pools reliably.

## Tests

Tests are grouped by domain inside `tests/unit/` and `tests/integration/`: database, embeddings, processing, storage, and operations. Database integration fixtures create uniquely named temporary databases and delete them afterward; they never read the application's database configuration.

```sh
env/bin/python -m pytest tests/unit -q
# Point only at a disposable PostgreSQL server with pgvector and an admin role.
export RYPIQ_TEST_ADMIN_URL='postgresql+psycopg://tester:password@localhost:5432/postgres'
env/bin/python -m pytest tests -q
```

Without that test URL, database integrations are skipped. Model integration tests require the local model above or `EMBEDDING_MODEL_PATH`; absent models are skipped, never downloaded. A skipped integration is not evidence of verified support. See [validation results](docs/VALIDATION.md) for the recorded TDD iterations and actual results.
