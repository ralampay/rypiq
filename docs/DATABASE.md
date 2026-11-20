# Database

RyPiq uses PostgreSQL with pgvector for application configuration, document persistence, and semantic search. See the [README](../README.md) for installation and ingestion examples, and [execution paths](EXECUTION_PATHS.md) for CLI and Python composition.

## Setup and configuration

See the [copyable configuration examples](../README.md#configuration-examples) for database, storage, model catalog, and shell environment setup.

Install PostgreSQL and the pgvector server extension on the database host. The Python `pgvector` package does not install that server extension. Run the commands below from the project root after installing RyPiq.

Copy [config/database.yml.dist](../config/database.yml.dist) to `config/database.yml`, then edit it for your host, port, database names, and PostgreSQL user. The local file is ignored by Git. Set `RYPIQ_DATABASE_PASSWORD` in the project-root `.env`, or export it in your shell. RyPiq preloads `.env` before configuration reads in both the CLI and Python loaders; existing environment values take precedence. The file is optional. Use `rypiq --doctor` to check configuration. The default configuration uses PostgreSQL's `postgres` database for maintenance connections. The maintenance and target databases must differ.

```sh
cp config/database.yml.dist config/database.yml
export DATABASE_ENV=development
export RYPIQ_DATABASE_PASSWORD='your-local-password'
rypiq db:create
```

The configured role needs permission to create/drop its databases and initialize the `vector` extension. Provision the extension with an administrator if the application role cannot create it. Errors after database creation leave the database in place: correct the permissions or extension installation and retry `db:migrate`.

`DATABASE_ENV` selects `development`, `test`, or `production`. `RYPIQ_DATABASE_CONFIG` selects an alternative YAML file by absolute path, including when running outside the checkout or using a non-editable package installation. [.env.example](../.env.example) lists supported variables; the CLI does not automatically load `.env` files. Do not commit real credentials.

## Independent-provider upgrade

Applications persist non-null `embedding_provision_type` and `inference_provision_type` columns, each constrained to `local`, `openai`, or `llama-server`. The Python/CLI defaults are independently `local`; `inference_model` remains nullable. The application-level `provision_type` column and old API argument/CLI flag have been removed.

Stop callers and deploy matching code before running `rypiq db:migrate`. The explicit transactional upgrade recognizes the old shared-provider schema, copies its provider into both new columns, and removes the old column. Existing IDs, API keys, timestamps, model settings, JSON configuration, documents, and embeddings are preserved. Older supported document layouts first pass through their existing historical upgrades. A partial/unknown schema is rejected; failure rolls back all upgrade changes. Running migration again against the current schema preserves data.

Old and new application versions cannot operate concurrently against this changed schema. This is a deliberate upgrade, not migration autogeneration. Database reset is unnecessary and would delete data.

Embedding rows retain their `provision_type` column, and processing snapshots retain the same key; both now correspond to the application's `embedding_provision_type`. Changing inference configuration does not invalidate embeddings. Changing embedding settings retains the existing reprocessing requirements.

After migration, use `applications:update` to select independent providers. See [updating provider configuration](PROVIDERS.md#updating-provider-configuration) for CLI and Python examples, JSON replacement behavior, credentials, verification, and reprocessing.

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

The explicit schema in [app/database/schema.py](../app/database/schema.py) declares
`applications`, `documents`, and `embeddings`. Applications retain their model,
provider, dimension, API-key, and JSONB configuration fields.

A `Document` is one original file: UUID, required name, application ownership, storage key,
filename, content type, source, JSONB metadata, and timestamps. An `Embedding`
belongs to a document and stores chunk `content`, vector, chunk index, model
snapshot (`model_name`, `embedding_model`, `embedding_size`, `provision_type`),
JSONB metadata, and timestamps. Parent deletion cascades to embeddings; deleting
an embedding keeps its parent. All deletion commands preserve stored files.
Applications cannot be deleted while documents exist, including empty parents.

Vectors support different sizes across applications. The configuration default is
768; valid sizes are 1–16000. Writes and searches validate finite, nonzero float32
vectors. The database checks vector dimensions against each embedding's snapshot.
Search joins through the parent for application scope and performs exact cosine
ordering with embedding-ID tie-breaking. Metadata filters use parameterized JSONB
containment before ranking, backed by a GIN metadata index. No approximate vector
index is created. Foreign-key lookup columns are indexed.

Document names are trimmed and nonempty. Fresh setup and migration install the
named `UNIQUE(application_id, name)` constraint and an empty-name check on the
non-null text column. Uniqueness is case-sensitive; applications may reuse names.
Document service/repository updates may rename a document under the same rules.
Name conflicts raise `DuplicateDocumentNameError`, including concurrent writes. Other
constraint failures raise `DatabaseIntegrityError`; adapter failures use `DatabaseError`
subclasses. See [exception contracts](EXCEPTIONS.md).

### Existing database upgrade

Back up the database and schedule a maintenance window before `db:migrate`.
The explicit upgrade locks the old documents table and runs in one transaction.
It recognizes the application-owned chunk schema and the older fixed-vector
schema. The latter first receives legacy applications grouped by `model_name`;
empty model names remain flagged for manual configuration.

Old chunk IDs become embedding IDs. New parent UUIDs are generated by grouping
rows by application, non-null `source_id`, and matching original-file attributes.
Rows without `source_id` get separate parents; conflicting file attributes split
groups. Every chunk retains its vector, text, model snapshot, metadata, and
timestamps. Parents retain file fields, shared file metadata, and original
`source_id` in metadata. Their timestamps span their children's creation/update
times. Stored files are never read or modified by this upgrade.

Existing parent-document databases receive names without changing their IDs or
timestamps. Unique trimmed filenames become names. Duplicate filenames receive
`filename (UUID)` names; empty filenames receive `Document UUID`. Unique filenames
are reserved first, then generated names are assigned in UUID order with numeric
suffixes if necessary. Allocation is per application. Both older chunk schemas
also receive these names during their upgrade. The name column, backfill, and
constraints are committed together; failures roll back all changes.

The upgrade checks row counts and validates the resulting schema before commit.
Failure restores the old schema and data. Unknown/partially modified schemas and
external dependencies that prevent replacing the old table are rejected.
Repeated migration is safe. `db:create` uses the same initializer on an existing
database. There is no migration discovery or revision journal.

Schema validation rejects `NOT VALID` constraints. Repair invalid data and
explicitly validate those constraints before retrying. Repository interfaces use
parameterized SQL and are intended for trusted callers, not authentication.

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

`applications:show APPLICATION_UUID` returns application settings without its API key. Creation also accepts `--embedding-provision-type`, `--inference-provision-type`, `--inference-model`, and `--data '{"purpose":"example"}'`. Embedding and inference execution support all three providers. See [Model providers](PROVIDERS.md) for endpoints, credentials, model aliases, and chat commands.

`applications:list` lists every application in UUID order. Its default text table
shows ID, name, embedding provider, inference provider, embedding model, embedding size, and inference
model. Missing or empty models display `-`; complete values are shown with control
characters escaped. It accepts only `--output text` or `--output json`; JSON
contains `id`, `name`, `embedding_provision_type`, `inference_provision_type`, `embedding_model`, `embedding_size`, and
`inference_model`, preserving saved values and using `null` for an unset inference
model. Empty results produce headers in text or `[]` in JSON. There are no CLI
pagination options; Python service and repository listing retain pagination.

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

Use `documents:create PATH --application-id UUID --name NAME` to save an original and generate
its embeddings. Required `--name` supplies the document reference.
`documents:list --application-id UUID` shows every ID, name, status, and current embedding
count; use `--output json` for structured summaries. `documents:show --id UUID` returns parent fields and ordered chunk
summaries; `documents:delete --id UUID` removes the parent and all chunks.
`embeddings:show` and `embeddings:delete` address individual chunks by `--id`.
A missing show fails; an absent delete returns `{"deleted": false}`.

Document create/process/show and embedding show support JSON export with `--output`.
Embedding show alone supports `--embeddings-full`; its exports always contain
complete vectors. Creation export failures report the persisted parent ID and
preserve the records. See the [CLI reference](../README.md#embedding-commands).

## Document status and processing

`Document.status` defaults to `pending`; allowed values are `pending`, `processing`,
`error`, and `done`, enforced by a PostgreSQL check constraint. `db:migrate` upgrades
recognized pre-status schemas transactionally, assigning `done` to parents with
embeddings and `pending` otherwise. It preserves IDs, timestamps, originals, and chunks.
Existing completed embeddings receive a snapshot of their application's settings.

Use `DocumentService.list(status='error', limit=100, offset=0)` for scoped fetching,
`DocumentRepository.list(status='pending')` for cross-application polling, or
`documents:list --application-id UUID --status error` from the CLI. An omitted filter
returns every status. These are reads, not work-claim operations.

`ProcessDocument` holds a session advisory lock for the document across short committed
transactions. Concurrent processors receive `ConflictError`; crashed processors release
their lock when the connection closes. Processing may be retried from any status.
`replace_embeddings()` locks the application before the document and verifies captured
processing settings before committing new chunks and `done` together.

The reserved document metadata key `_rypiq_processing` stores non-secret model,
dimension, provider, preprocessing, and chunking settings. Repository updates preserve
this key; successful processing replaces it. Search checks exact embedding compatibility,
independently of document status, and guards vector dimensions before cosine evaluation.
Chunking changes do not invalidate existing vectors. Configuration changes during
processing abort replacement; old embeddings remain available for inspection.

## CRUD services

`ApplicationService`, `DocumentService`, and `EmbeddingService` accept caller-owned repositories and expose `create`, `get`, `list`, `update`, and `delete`. They return domain dataclasses, lists, or deletion booleans and raise exceptions; they never print, load configuration, generate embeddings, or close borrowed resources. Existing application CLI operations use the same service layer.

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
    document = documents.create(Document(filename='example.txt', name='Example document'))
    document = documents.get(document.id)
    documents.update(replace(document, metadata={'reviewed': True}))
    page = documents.list(limit=10, offset=0)
    applications.update(replace(application, name='Renamed'))
    documents.delete(document.id)
    applications.delete(application.id)
```

Document service creation assigns application ownership without mutating the input.
Embedding service calls take an `Embedding` with a parent `document_id` and model
snapshots matching that parent's application. Both services scope their calls by
application. Embedding lists require a document ID; document lists use pagination.
CRUD does not generate vectors or touch object storage. Updates take complete
objects; parent ownership, chunk parent IDs, and model snapshots are immutable.

Missing service reads and repository/service updates raise the appropriate
`ApplicationNotFoundError`, `DocumentNotFoundError`, or `EmbeddingNotFoundError`
(all compatible with `KeyError`). Repository reads return `None`; missing deletes
return `False`. Invalid inputs raise `ValidationError` subclasses (compatible with
`ValueError`); conflicts have distinct `ConflictError` subclasses. Document deletion cascades to chunks and preserves the
stored original. Creation commits a pending parent before attaching its stored original.
Processing commits `processing`, then atomically replaces chunks and marks `done`.
Failure marks `error` while preserving existing chunks and the original. A failure to
attach a newly stored original compensates that object; filesystem and database
operations are not one transaction. The lower-level `create_with_embeddings()` API
remains available for callers supplying a complete aggregate.

Application updates preserve API keys and creation timestamps. Embedding model, size, embedding provision type, and preprocessing settings can change while documents exist. Reprocess each affected document with `documents:process --document-id UUID`. Search excludes incompatible embedding snapshots until replacement succeeds. Application deletion is restricted until its documents have been removed. Application deletion and document creation use transaction locks to enforce ownership and deletion restrictions. API key rotation and document updates remain outside the CLI; `embeddings:show` and `embeddings:delete` access individual chunk records by UUID. `applications:update` also resolves model catalog aliases; see [Model providers](PROVIDERS.md).

## Database integration tests

Database integration fixtures create uniquely named temporary databases and delete them afterward; they never read the application's database configuration. Point `RYPIQ_TEST_ADMIN_URL` only at a disposable PostgreSQL server with pgvector and an admin role:

```sh
export RYPIQ_TEST_ADMIN_URL='postgresql+psycopg://tester:password@localhost:5432/postgres'
env/bin/python -m pytest tests/integration/database -q
```

Without that URL, database integrations are skipped. A skip is not evidence of verified support. See [validation results](VALIDATION.md) for historical runs; run the tests again to validate current changes.

## Durable document processing queue

`db:create` and `db:migrate` include the explicit `document_processing_jobs` table.
It stores document ownership, batch size, queue state, attempts, a claim token and
lease expiry, timestamps, and a safe error message. The document foreign key cascades
only when document deletion is allowed; repository deletion rejects active jobs.
A partial unique index enforces one queued/running job per document. Migration
validates this index as well as the table shape and rejects incompatible definitions.

Worker claims use `FOR UPDATE SKIP LOCKED`; token checks fence writes after claim
expiry. Job success and embedding replacement are one database transaction. No
migration files, external broker, or extra database backend are introduced.
See [API queue guarantees](API.md#queue-and-recovery-guarantees).
