# Database

RyPiq uses PostgreSQL with pgvector for application configuration, document persistence, and semantic search. See the [README](../README.md) for installation and ingestion examples, and [execution paths](EXECUTION_PATHS.md) for CLI and Python composition.

## Setup and configuration

See the [copyable configuration examples](../README.md#configuration-examples) for database, storage, model catalog, and shell environment setup.

Install PostgreSQL and the pgvector server extension on the database host. The Python `pgvector` package does not install that server extension. Run the commands below from the project root after installing RyPiq.

Copy [config/database.yml.dist](../config/database.yml.dist) to `config/database.yml`, then edit it for your host, port, database names, and PostgreSQL user. The local file is ignored by Git. Export `RYPIQ_DATABASE_PASSWORD` in your shell. The default configuration uses PostgreSQL's `postgres` database for maintenance connections. The maintenance and target databases must differ.

```sh
cp config/database.yml.dist config/database.yml
export DATABASE_ENV=development
export RYPIQ_DATABASE_PASSWORD='your-local-password'
rypiq db:create
```

The configured role needs permission to create/drop its databases and initialize the `vector` extension. Provision the extension with an administrator if the application role cannot create it. Errors after database creation leave the database in place: correct the permissions or extension installation and retry `db:migrate`.

`DATABASE_ENV` selects `development`, `test`, or `production`. `RYPIQ_DATABASE_CONFIG` selects an alternative YAML file by absolute path, including when running outside the checkout or using a non-editable package installation. [.env.example](../.env.example) lists supported variables; the CLI does not automatically load `.env` files. Do not commit real credentials.

## Database commands

| Command | Behavior |
| --- | --- |
| `db:create` | Create the database if missing, enable pgvector, and create the fixed application tables. Repeating it preserves data. |
| `db:migrate` | Initialize missing tables, upgrade the recognized legacy documents schema, and validate managed tables. |
| `db:drop` | Delete the configured target database and its data. A missing target is a no-op. |
| `db:reset` | Drop the configured target database, then create it again with the extension and application tables. |

Commands return JSON. Failures return a nonzero exit code with a concise error. Database creation/drop use an autocommit maintenance connection; extension and table initialization run in one transaction. A failed reset after the drop cannot restore the old data. Active external connections make drop/reset fail; the commands do not forcibly terminate other sessions.

**Drop and reset delete database contents, including unrelated tables in that database.** They preserve original files in object storage. Close other applications' connections before resetting, and use a dedicated application database. Run schema initialization with one writer at a time.

## Fixed schema and embeddings

The explicit schema in [app/database/schema.py](../app/database/schema.py) declares `public.applications` and `public.documents`. Applications have a UUID, name, generated unique API key, embedding model, provision type (`local`, `openai`, `llama-server`), embedding size, optional inference model, JSONB `data`, and timestamps. API keys appear only in application creation output; list/show/update omit them. Keys are stored as supplied credentials, but authentication is not implemented.

Every persisted document belongs to an application and copies its `embedding_model`, `embedding_size`, and `provision_type`. Existing `model_name` remains the provider-reported identifier. Ownership and embedding snapshots cannot change through document updates. Applications cannot be deleted while documents reference them; application updates are available through `applications:update`; deletion is available through `applications:delete APPLICATION_UUID`.

The vector column supports different sizes across applications. `dimensions` in `database.yml` supplies the default size for new applications (**768**); it no longer fixes the table's vector size. Writes and queries require finite, nonzero float32 vectors matching the application's size (1–16000). Search requires an application ID, filters before cosine comparison, and preserves exact cosine ordering with UUID tie-breaking. No approximate vector index is created.

`db:migrate` explicitly upgrades the original fixed-vector documents schema in a single transaction under a table lock. Back up existing data before upgrading and schedule a maintenance window. The upgrade creates one legacy application per distinct `model_name`, preserves the actual vector size and all existing document values, and assumes `local` provisioning. Empty legacy model names remain empty and are flagged in application `data`; configured embedding operations reject these until explicitly repaired. Stored files are untouched. Repeating migration is a no-op. Unknown or partially modified schemas are rejected without changes; no migration discovery or revision journal is introduced. `db:create` uses the same initializer on an existing database.

CRUD uses parameterized SQLAlchemy Core queries. Python repository callers must provide application ownership and matching snapshots for document writes, and `application_id` for vector searches. These are persistence interfaces, not an authentication boundary.

Schema validation rejects PostgreSQL `NOT VALID` constraints even when their definitions match: existing rows have not been verified. Repair invalid rows and explicitly validate the constraint before retrying `db:migrate`; migration does not silently validate or rewrite application data.

## Application commands

Create an application before ingesting documents, then use its UUID with ingestion and search:

```sh
rypiq applications:create --name 'My application' \
  --embedding-model models/embeddinggemma-300m-Q4_0.gguf --embedding-size 768
rypiq applications:list
rypiq applications:list --output json
rypiq applications:show APPLICATION_UUID
rypiq applications:update APPLICATION_UUID --name 'Renamed application'
```

`applications:show APPLICATION_UUID` returns application settings without its API key. Creation also accepts `--provision-type`, `--inference-model`, and `--data '{"purpose":"example"}'`. Embedding and inference execution support all three providers. See [Model providers](PROVIDERS.md) for endpoints, credentials, model aliases, and chat commands.

`applications:list` returns every application's ID and name in UUID order. It
prints an `ID` / `NAME` text table by default and accepts only `--output text` or
`--output json`. Empty results produce headers in text or `[]` in JSON. There are
no CLI pagination options; Python service and repository listing retain pagination.

`applications:update APPLICATION_UUID --name NAME` overrides the name while
preserving omitted settings. It also supports the existing embedding, provider,
inference, and data options; see the [application option reference](../README.md#application-commands).
Show and update return JSON without the API key.

Delete an application only after its documents have been removed:

```sh
rypiq applications:delete APPLICATION_UUID
```

Deletion runs without a confirmation prompt and returns `{"deleted": true}` when
a record was removed or `{"deleted": false}` if it was already absent. Both exit
successfully. If documents still reference the application, deletion fails with a
nonzero exit and preserves the application and documents. Object-storage files
are never removed by this command.

All application CLI names use the plural `applications:*` domain. The former
singular names and `ls-applications` are no longer accepted.

## Embedding commands

`embeddings:create PATH --application-id UUID` saves the original file and creates
one row in `documents` per nonempty chunk. Model/provider settings come from the
application, as do `data.chunking.chunk_size` and `data.chunking.stride` (defaults
500 and 400). Configure them using `applications:create` or `applications:update`
with `--chunk-size` and `--stride`. Values must be positive integers with stride
no greater than chunk size. No schema migration is required. Updating chunking
settings leaves existing records and their metadata unchanged.

`embeddings:show --id UUID` reads one chunk; `embeddings:delete --id UUID` deletes
one chunk. Neither requires an application ID. These use the repository's globally
unique record IDs, not the shared `source_id`, and do not provide an authentication
boundary. Deletion never removes stored original files, including after the last
chunk is deleted. An absent deletion returns `{"deleted": false}` successfully;
an absent show exits nonzero.

Create and show support `--output PATH` to export full JSON after a successful
operation, while still printing a console response. Console vectors are shortened
unless `--embeddings-full` is supplied; exported vectors are always complete.
Export failure after creation preserves database records and storage files and
reports the persisted chunk IDs. See the [embedding CLI reference](../README.md#embedding-commands)
for response fields, options, and retry examples. There is no embedding update CLI.
The former `embed` and `ingest` commands are replaced by `embeddings:create`.

## CRUD services

`ApplicationService` and `DocumentService` accept caller-owned repositories and expose `create`, `get`, `list`, `update`, and `delete`. They return domain dataclasses, lists, or deletion booleans and raise exceptions; they never print, load configuration, generate embeddings, or close borrowed resources. Existing application CLI operations use the same service layer.

```python
from dataclasses import replace
from app.database.factory import get_database
from app.models.application import Application
from app.models.document import Document
from app.services.application import ApplicationService
from app.services.document import DocumentService

with get_database() as database:
    applications = ApplicationService(repository=database.applications())
    application = applications.create(Application(
        name='Example', embedding_model='example-model', embedding_size=2,
    ))
    documents = DocumentService(
        application_id=application.id,
        repository=database.documents(), applications=database.applications(),
    )
    document = documents.create(Document(content='Example text', embedding=[1.0, 0.0]))
    document = documents.get(document.id)
    documents.update(replace(document, metadata={'reviewed': True}))
    page = documents.list(limit=10, offset=0)
    applications.update(replace(application, name='Renamed'))
    documents.delete(document.id)
    applications.delete(application.id)
```

Document creation copies application embedding settings without mutating the input object. Supply vectors compatible with that application's model; CRUD does not re-embed content. Updates take a complete dataclass, not a partial patch. Document ownership and embedding snapshots remain immutable. Every document service read/write/list is scoped to its application in SQL; repository APIs retain optional unscoped access for trusted library callers.

Missing reads/updates raise `KeyError`. Deletes return `False` for missing records (including documents outside the service's scope); lists return an empty list when no records match. Invalid inputs raise `ValueError` before writes. Document deletion removes only the database record and preserves object-storage files, which can be shared by multiple chunks.

Application updates preserve API keys and creation timestamps. Embedding model, size, provision type, and preprocessing settings can change only when no documents reference the application. Other settings can be updated while documents exist. Application deletion is restricted until its documents have been removed. These rules execute under transaction locks so document creation cannot race past them. API key rotation and document updates remain outside the CLI; `embeddings:show` and `embeddings:delete` access individual chunk records by UUID. `applications:update` also resolves model catalog aliases; see [Model providers](PROVIDERS.md).

## Database integration tests

Database integration fixtures create uniquely named temporary databases and delete them afterward; they never read the application's database configuration. Point `RYPIQ_TEST_ADMIN_URL` only at a disposable PostgreSQL server with pgvector and an admin role:

```sh
export RYPIQ_TEST_ADMIN_URL='postgresql+psycopg://tester:password@localhost:5432/postgres'
env/bin/python -m pytest tests/integration/database -q
```

Without that URL, database integrations are skipped. A skip is not evidence of verified support. See [validation results](VALIDATION.md) for historical runs; run the tests again to validate current changes.
