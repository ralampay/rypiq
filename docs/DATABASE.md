# Database

RyPiq uses PostgreSQL with pgvector for application configuration, document persistence, and semantic search. See the [README](../README.md) for installation and ingestion examples, and [execution paths](EXECUTION_PATHS.md) for CLI and Python composition.

## Setup and configuration

Install PostgreSQL and the pgvector server extension on the database host. The Python `pgvector` package does not install that server extension. Run the commands below from the project root after installing RyPiq.

Edit [config/database.yml](../config/database.yml) for your host, port, database names, and PostgreSQL user. Export `RYPIQ_DATABASE_PASSWORD` in your shell. The default configuration uses PostgreSQL's `postgres` database for maintenance connections. The maintenance and target databases must differ.

```sh
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

The explicit schema in [app/database/schema.py](../app/database/schema.py) declares `public.applications` and `public.documents`. Applications have a UUID, name, generated unique API key, embedding model, provision type (`local`, `openai`, `llama-server`), embedding size, optional inference model, JSONB `data`, and timestamps. API keys appear only in application creation output; list/show omit them. Keys are stored as supplied credentials, but authentication is not implemented.

Every persisted document belongs to an application and copies its `embedding_model`, `embedding_size`, and `provision_type`. Existing `model_name` remains the provider-reported identifier. Ownership and embedding snapshots cannot change through document updates. Applications cannot be deleted while documents reference them; application editing/deletion commands are not included.

The vector column supports different sizes across applications. `dimensions` in `database.yml` supplies the default size for new applications (**768**); it no longer fixes the table's vector size. Writes and queries require finite, nonzero float32 vectors matching the application's size (1–16000). Search requires an application ID, filters before cosine comparison, and preserves exact cosine ordering with UUID tie-breaking. No approximate vector index is created.

`db:migrate` explicitly upgrades the original fixed-vector documents schema in a single transaction under a table lock. Back up existing data before upgrading and schedule a maintenance window. The upgrade creates one legacy application per distinct `model_name`, preserves the actual vector size and all existing document values, and assumes `local` provisioning. Empty legacy model names remain empty and are flagged in application `data`; configured embedding operations reject these until explicitly repaired. Stored files are untouched. Repeating migration is a no-op. Unknown or partially modified schemas are rejected without changes; no migration discovery or revision journal is introduced. `db:create` uses the same initializer on an existing database.

CRUD uses parameterized SQLAlchemy Core queries. Python repository callers must provide application ownership and matching snapshots for document writes, and `application_id` for vector searches. These are persistence interfaces, not an authentication boundary.

## Application commands

Create an application before ingesting documents, then use its UUID with ingestion and search:

```sh
rypiq application:create --name 'My application' \
  --embedding-model models/embeddinggemma-300m-Q4_0.gguf --embedding-size 768
rypiq application:list
rypiq application:show APPLICATION_UUID
```

`application:show APPLICATION_UUID` returns application settings without its API key. Creation also accepts `--provision-type`, `--inference-model`, and `--data '{"purpose":"example"}'`. Only local embedding execution is implemented; remote configurations can be stored but ingestion/search reject them before effects.

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

Application updates preserve API keys and creation timestamps. Embedding model, size, and provision type can change only when no documents reference the application. Other settings can be updated while documents exist. Application deletion is restricted until its documents have been removed. These rules execute under transaction locks so document creation cannot race past them. API key rotation and new CRUD CLI commands are outside this interface.

## Database integration tests

Database integration fixtures create uniquely named temporary databases and delete them afterward; they never read the application's database configuration. Point `RYPIQ_TEST_ADMIN_URL` only at a disposable PostgreSQL server with pgvector and an admin role:

```sh
export RYPIQ_TEST_ADMIN_URL='postgresql+psycopg://tester:password@localhost:5432/postgres'
env/bin/python -m pytest tests/integration/database -q
```

Without that URL, database integrations are skipped. A skip is not evidence of verified support. See [validation results](VALIDATION.md) for historical runs; run the tests again to validate current changes.
