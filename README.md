# RyPiq

A Python CLI and library for processing text, generating embeddings and text responses, storing original files, and searching documents with PostgreSQL and pgvector.

## Setup

Use Python 3.14.3 or newer. Persistence and search require PostgreSQL with pgvector; see [database prerequisites](docs/DATABASE.md#setup-and-configuration).

```sh
python -m venv env
. env/bin/activate
python -m pip install -e '.[dev]'
```

Start with the [configuration examples](#configuration-examples) below, then initialize PostgreSQL using the [database setup guide](docs/DATABASE.md#setup-and-configuration).

## Configuration examples

Run these examples from the project root. You need PostgreSQL with pgvector and
an existing embedding model; RyPiq does not download models.

### Database: `config/database.yml`

Copy the [distribution template](config/database.yml.dist) to your local configuration, which Git ignores:

```sh
cp config/database.yml.dist config/database.yml
```

```yaml
development:
  adapter: postgresql
  host: localhost
  port: 5432
  user: rypiq
  password_env: RYPIQ_DATABASE_PASSWORD
  database: rypiq_development
  maintenance_database: postgres
  dimensions: 768
```

RyPiq loads the project-root `.env` before reading configuration, for both CLI
commands and direct Python configuration calls. For example:

```dotenv
DATABASE_ENV=development
RYPIQ_DATABASE_PASSWORD=replace-with-your-local-password
```

Existing process environment variables take precedence, including empty values.
The `.env` file is optional and Git-ignored. Its location follows the project's
source root, not the working directory; parent directories are not searched.
Loaded values stay in the process environment, so restart the process to pick up
changes to those values. Check your setup with `rypiq --doctor`.

Replace the host, user, and database for your setup. `dimensions` is the default
embedding size for new applications; choose a size supported by your model. See [database prerequisites](docs/DATABASE.md#setup-and-configuration)
for PostgreSQL permissions and pgvector setup.

### File storage: `config/storage.yml`

Copy the [distribution template](config/storage.yml.dist) to your local configuration, which Git ignores:

```sh
cp config/storage.yml.dist config/storage.yml
```

```yaml
development:
  adapter: local
  root: storage/objects/development
```

Original files go here; text and vectors go into PostgreSQL. With the standard
`config/` layout, this path is relative to the project root. Use an absolute path
if your configuration lives elsewhere.

### Model aliases: `config/models.yml` (optional)

```yaml
models:
  document-embedding:
    provider: local
    capabilities: [embedding]
    path: ../models/embeddinggemma-300m-Q4_0.gguf
```

Model paths are relative to this YAML file. Use `@document-embedding` instead of
the model path in commands. Skip this file if you use a direct model path.

### Create an application and embed a document

```sh
export DATABASE_ENV=development
export RYPIQ_DATABASE_PASSWORD='replace-with-your-local-password'

rypiq db:create
rypiq applications:create --name 'Document search' \
  --embedding-model models/embeddinggemma-300m-Q4_0.gguf --embedding-size 768 \
  --chunk-size 500 --stride 400

# Replace APPLICATION_UUID with the ID returned above:
rypiq documents:create --name "Example document" /path/to/document.pdf --application-id APPLICATION_UUID
```

Model and chunking settings are saved on the application. Chunk size and stride
default to 500 and 400; change them with `applications:update`.

`DATABASE_ENV` defaults to `development`; `STORAGE_ENV` follows it unless set
separately. The CLI does not load `.env` automatically. To use files outside the
standard layout, set `RYPIQ_DATABASE_CONFIG`, `RYPIQ_STORAGE_CONFIG`, or
`RYPIQ_MODELS_CONFIG` to their absolute paths.

See [.env.example](.env.example) for environment variables,
[application options](#application-commands) for updates, and
[model providers](docs/PROVIDERS.md) for remote models and advanced settings.

### Check your configuration

```sh
rypiq --doctor
```

Doctor checks database/storage YAML, selected environments, required credential
variables, storage paths, and optional model catalog files. When database configuration
is valid, it also connects to PostgreSQL and checks pgvector and required tables/columns. It prints `COMPLETE`,
`REQUIRED`, `INVALID`, or `OPTIONAL` beside each item, with a suggested fix when
needed. Credential values are never printed.

Exit code `0` means required configuration and database checks passed; `1` means
a required check failed. Optional catalog/model issues do not change
the exit code. Use `--doctor` alone, without another command.

Database inspection is read-only, with a 5-second connection timeout and a
5-second timeout per query. Doctor never creates databases, runs migrations,
loads models, or contacts remote model providers. It does not validate table
constraints, column types, database write permissions, or saved application
settings. Filesystem permission checks do not guarantee successful writes.

## Logging

Normal CLI commands write to `logs/<environment>.log`, following `LOG_ENV`, then
`DATABASE_ENV`, then `development`. Defaults are readable INFO logs; YAML,
environment variables, and Python initialization can override them. See
[logging configuration and callable components](docs/LOGGING.md).

## Database

See [Database documentation](docs/DATABASE.md) for configuration, lifecycle commands, schema and migration behavior, application management, CRUD services, and database integration tests.

## Model providers

See [Model providers](docs/PROVIDERS.md) for local GGUF execution, existing llama.cpp HTTP servers, OpenAI, the model catalog, chat, and model discovery.

## Ingest and search

Provide a compatible local GGUF embedding model; these commands do not download one.

```sh
rypiq applications:create --name 'My application' \
  --embedding-model models/embeddinggemma-300m-Q4_0.gguf --embedding-size 768
rypiq applications:list
# Use the UUID returned by applications:create:
rypiq documents:create --name "Example document" samples/sample.txt --application-id APPLICATION_UUID
rypiq search 'What is this document about?' --application-id APPLICATION_UUID \
  --model models/embeddinggemma-300m-Q4_0.gguf
```

See [application commands](docs/DATABASE.md#application-commands) for configuration options and supported providers.

`documents:create` uses the application's saved model and chunking settings. Search also uses the application model and supports a compatible local `--model` path override. The former `embed` and `ingest` CLI commands have been removed; standalone embedding remains available through the Python API.

`config/storage.yml` controls original-file storage. `STORAGE_ENV` defaults to `DATABASE_ENV`; `RYPIQ_STORAGE_CONFIG` overrides its configuration path. Storage paths remain independent of database reset.

Library callers should use `with create_context() as context:` from `app.context` or `with get_database() as database:` from `app.database.factory` to release connection pools reliably.
See [Python exceptions](docs/EXCEPTIONS.md) for the public hierarchy, compatibility, and recovery examples.


## CLI reference

Commands return JSON except `applications:list`, which defaults to a text table and supports `--output json`. The standalone `--doctor` option prints a text report. Failures exit nonzero with a diagnostic. Use `rypiq --help` to list commands and `rypiq COMMAND --help` for individual usage. `-h` is also supported. Run the examples from the project root and replace `APPLICATION_UUID` with an existing application's UUID. Flag spelling matters: chunking and local runtime flags use underscores, while application and chat flags use hyphens.

- [Database](#database-commands)
- [Applications](#application-commands)
- [Text processing](#text-processing)
- [Embeddings](#embedding-commands)
- [Documents and search](#document-ingestion-and-search)
- [Inference](#inference-commands)
- [Model discovery](#model-discovery)
- [Object storage](#object-storage)

### Database commands

These commands take no positional arguments or command-specific flags. They use the selected database configuration; see the [database setup guide](docs/DATABASE.md#setup-and-configuration) for environment variables and permissions.

| Command | Behavior |
| --- | --- |
| `rypiq db:create` | Create the database if missing, enable pgvector, and initialize the fixed tables. |
| `rypiq db:migrate` | Create missing tables, upgrade the recognized legacy schema, and reject incompatible schemas. |
| `rypiq db:drop` | Delete the configured database and all its contents. |
| `rypiq db:reset` | Delete and recreate the configured database and fixed tables. |

**`db:drop` and `db:reset` delete database data, including unrelated tables in that database.** Original files in object storage remain intact.

```sh
rypiq db:create
rypiq db:migrate
```

### Application commands

An application independently selects embedding and inference providers and models. Create and update accept local GGUF paths, remote model IDs, or compatible `@catalog-alias` references. See [provider configuration](docs/PROVIDERS.md#application-provider-configuration) for endpoints and credential environment references.

| Command | Required arguments | Purpose |
| --- | --- | --- |
| `rypiq applications:create` | `--name NAME --embedding-model MODEL` | Create an application and generate its API key. |
| `rypiq applications:list` | None | List every application's ID, name, model configuration, and document count; `--output text` (default) or `--output json`. |
| `rypiq applications:show APPLICATION_UUID` | Application UUID positional argument | Show one application's details. |
| `rypiq applications:delete APPLICATION_UUID` | Application UUID positional argument | Delete an application that has no documents. |
| `rypiq applications:update APPLICATION_UUID` | Application UUID positional argument | Update supplied fields; omitted fields remain unchanged. |

Creation returns the generated API key. List, show, and update omit it.

`applications:delete APPLICATION_UUID` returns `{"deleted": true}` when deleted or `{"deleted": false}` when already absent. Both deletion outcomes exit successfully. Applications with documents cannot be deleted and return a nonzero exit; stored files are never removed by this command. Deletion runs without an interactive prompt.

Application CLI commands use the plural `applications:*` domain. The former
`application:*` names and `ls-applications` are no longer supported.

`applications:list` lists all applications in UUID order, without pagination arguments.
The default text table has `ID`, `NAME`, `EMBEDDING PROVIDER`, `INFERENCE PROVIDER`, `EMBEDDING MODEL`,
`EMBEDDING SIZE`, `INFERENCE MODEL`, and `DOCUMENTS` columns. Model values are the saved
configuration, without alias resolution; missing or empty models display `-`.
Columns expand to show complete values. Control characters in text cells are
escaped so each application occupies one row. An empty database prints only the
headers. JSON output contains `id`, `name`, `embedding_provision_type`, `inference_provision_type`, `embedding_model`,
`embedding_size`, `inference_model`, and integer `document_count`, returning `[]` when empty. It preserves
original string values, integer embedding sizes, and `null` for an unset inference model. Document counts include parent documents, not embedding chunks, and are zero for applications without documents. Listing does not guarantee a snapshot during concurrent changes.

```sh
rypiq applications:list
rypiq applications:list --output text
rypiq applications:list --output json
```

Example JSON output:

```json
[
  {
    "id": "00000000-0000-0000-0000-000000000001",
    "name": "My application",
    "embedding_provision_type": "local",
    "inference_provision_type": "local",
    "embedding_model": "my-embedding-model",
    "embedding_size": 768,
    "inference_model": null,
    "document_count": 0
  }
]
```

| Create/update option | Creation default or requirement | Update behavior |
| --- | --- | --- |
| `--name NAME` | Required | Replace the name. |
| `--embedding-model MODEL` | Required | Replace the embedding model. |
| `--chunk-size N` | `500` when absent from application data | Replace saved maximum characters per chunk. |
| `--stride N` | `400` when absent from application data | Replace saved character advance; must be positive and no greater than chunk size. |
| `--embedding-size N` | Database configuration's `dimensions` (normally `768`); range `1–16000` | Replace the embedding size. |
| `--embedding-provision-type TYPE` | `local`; choices: `local`, `openai`, `llama-server` | Replace only the embedding provider. |
| `--inference-provision-type TYPE` | Independently defaults to `local`; same choices | Replace only the inference provider. |
| `--inference-model MODEL` | Unset | Set the inference model. |
| `--data JSON` | `{}`; must be a JSON object | Replace the entire data object, not a partial merge. |
| `--clear-inference-model` | Not available on create | Clear the inference model; mutually exclusive with `--inference-model`. |

`--chunk-size` and `--stride` are saved as `data.chunking.chunk_size` and `data.chunking.stride`. Explicit flags override those values in `--data`; without `--data`, updates preserve other application data. Missing settings default independently to 500 and 400. Chunking changes affect future uploads and are allowed when documents already exist.

Embedding model, size, provider, and embedding preprocessing settings can change while documents exist. Reprocess each document to regenerate its embeddings; search excludes embeddings incompatible with the current settings. See [database documentation](docs/DATABASE.md) for persistence rules.

| List option | Default | Meaning |
| --- | --- | --- |
| `--output text\|json` | `text` | Format for all application IDs and names. |

```sh
rypiq applications:create --name 'My application' \
  --embedding-model models/embeddinggemma-300m-Q4_0.gguf --embedding-size 768 \
  --inference-model models/your-chat-model.gguf --data '{"purpose":"local search"}'
rypiq applications:list --output json
rypiq applications:show APPLICATION_UUID
rypiq applications:update APPLICATION_UUID --name 'Renamed application'
rypiq applications:update APPLICATION_UUID --clear-inference-model
rypiq applications:delete APPLICATION_UUID
```

Replace `models/your-chat-model.gguf` with your existing inference model. Models are not downloaded by these commands.

### Updating existing applications

When upgrading from the shared-provider schema, stop application callers, deploy the new code, and run `rypiq db:migrate` before restarting callers. Migration copies the previous provider into both new fields and preserves documents and embeddings. The former application `provision_type` field and `--provision-type` flag are removed; new applications default each provider independently to `local`.

Inspect the saved configuration, then change inference without changing embeddings:

```sh
rypiq applications:show APPLICATION_UUID
rypiq applications:update APPLICATION_UUID \
  --inference-provision-type openai --inference-model YOUR_OPENAI_MODEL_ID
rypiq applications:show APPLICATION_UUID
rypiq chat 'What does the guide say about setup?' --application-id APPLICATION_UUID
```

Set `OPENAI_API_KEY` in the environment before chat. OpenAI's default endpoint and credential reference apply when no explicit inference options are saved. If switching from another remote endpoint, replace stale inference options as described in [updating provider configuration](docs/PROVIDERS.md#updating-provider-configuration). Inference-only changes require no reprocessing.

Omitted update flags retain their saved values. `--data` replaces the entire JSON object, including embedding options, chunking, and application metadata; it does not merge nested keys. See the provider guide for complete JSON and Python update examples, local/llama-server configuration, and embedding reprocessing instructions.

### Text processing

| Syntax | Positional arguments | Behavior |
| --- | --- | --- |
| `rypiq process text CONTENT` | Literal text, quoted if it contains spaces | Return text, character count, and word count. |
| `rypiq process text-file PATH --chunk_size N --stride N` | Supported document path | Split a file into chunks and return their text and offsets. |

| Option | Requirement | Meaning |
| --- | --- | --- |
| `--chunk_size N` | Required for `text-file`; rejected for `text` | Maximum characters per chunk; positive integer. |
| `--stride N` | Required for `text-file`; rejected for `text` | Characters to advance; positive integer no greater than chunk size. |

```sh
rypiq process text 'Hello world'
rypiq process text-file samples/sample.txt --chunk_size 500 --stride 400
```

### Supported document files

Processing, embedding, and ingestion accept `.txt`, `.pdf`, `.docx`, `.xlsx`, `.xls`, `.xlsm`, and `.pptx` (case-insensitive). The existing `text-file` CLI argument and Python entrypoint names apply to all these formats.

| Format | Extracted content |
| --- | --- |
| TXT | UTF-8 text, unchanged. |
| PDF | Embedded page text in page order; no OCR for scans or images. |
| DOCX | Body paragraphs and tables, including nested tables, in document order. |
| XLSX / XLSM / XLS | All worksheets, including hidden sheets, in workbook order, with worksheet names. |
| PPTX | Slide text and tables, including grouped shapes, in slide order and stored shape order. |

Paragraphs and rows are separated by newlines and table cells by tabs. Empty rows and trailing empty cells are omitted; interior empty cells retain their positions. Spreadsheet dates use ISO format, booleans use `true`/`false`, and numbers use their underlying values rather than Excel display formatting. Empty worksheets contribute no text.

Excel formulas use saved results only: calculate and save the workbook in Excel before ingestion. Missing cached results are omitted. Macros are never executed, formulas are not evaluated, and external links are not refreshed. Word headers, footers, comments and floating text boxes, PowerPoint speaker notes and chart data, and text inside images are excluded. PDF extraction follows the parser's text order and does not reconstruct complex visual layouts.

Offsets refer to the extracted text, not positions within the original binary file. Processing can return zero chunks; embedding and ingestion reject files without usable text. Encrypted and malformed documents fail with a format-specific error. Ingestion stores the original file unchanged.

Install the project dependencies to obtain the document readers; no separate extraction service or model is needed.

```sh
rypiq process text-file report.pdf --chunk_size 500 --stride 400
rypiq documents:create --name "Presentation" presentation.pptx --application-id APPLICATION_UUID
rypiq documents:create --name "Workbook" workbook.xlsx --application-id APPLICATION_UUID
```

### Embedding commands

Documents represent original files; embeddings contain their searchable text chunks.

| Command | Behavior |
| --- | --- |
| `rypiq documents:create PATH --application-id UUID --name NAME` | Insert a pending document, store its original, then process it synchronously. |
| `rypiq documents:process --document-id UUID` | Reprocess the stored original using current application settings. |
| `rypiq documents:list --application-id UUID [--status STATUS] [--output FORMAT]` | List document IDs, names, statuses, and embedding counts. |
| `rypiq documents:show --id UUID` | Show parent fields and ordered chunk summaries. |
| `rypiq documents:delete --id UUID` | Delete the parent and all its embeddings; preserve the original file. |
| `rypiq embeddings:show --id UUID` | Show one chunk with text, vector, model information, JSON metadata, and `document_id`. |
| `rypiq embeddings:delete --id UUID` | Delete one chunk; preserve its parent, siblings, and original file. |

Creation requires `--name NAME`. Names are trimmed, nonempty, and unique within
an application; uniqueness is case-sensitive (`Report` and `report` are distinct).
Reusing a name in another application is allowed. Duplicate creation fails before
embedding where possible; a database constraint also handles concurrent requests.

`documents:list --application-id UUID` lists every document in UUID order, with
`ID`, `NAME`, `STATUS`, and `EMBEDDINGS` columns. Use `--output json` for an array of
`{id, name, status, embedding_count}` records. Counts reflect currently stored embeddings,
including zero for empty parents. Empty applications produce headers or `[]`;
a missing application fails. There are no CLI pagination options.

Creation uses saved application model/provider and chunking settings. It accepts
`--batch_size N` (positive integer, default 1). Configure chunking with
`applications:create` or `applications:update --chunk-size N --stride N`;
missing settings default independently to 500/400. Creation does not accept
per-command model or chunking overrides. Use a distinct name when creating another document in the same application.

Document create/process/show return the parent fields (including `status`), `embedding_count`, and `embeddings`
ordered by chunk index and ID. Each summary contains `id`, `chunk_index`, `content`,
and `metadata`; vectors are available through `embeddings:show`.
The parent stores file information and JSON metadata, not full extracted text.

Document create/process/show and embedding show accept `--output PATH` to export JSON.
Only embedding show accepts `--embeddings-full`: console vectors longer than six
values are abbreviated by default, while exports always contain complete vectors.
Export failure after creation preserves records and files and reports the document
ID. Retry with `documents:show --id UUID --output PATH`.

Documents progress through `pending` → `processing` → `done` or `error`.
Creation inserts the pending parent before storing or processing the file. Processing
failures retain the parent and stored original for retry; diagnostics include its UUID.
A storage failure before attachment can leave an error document without an original;
that record must be deleted and uploaded again. `documents:process` never falls back
to the old upload path.

```sh
rypiq documents:list --application-id APPLICATION_UUID --status error --output json
rypiq documents:process --document-id DOCUMENT_UUID --batch_size 8 --output processed.json
```

Reprocessing preserves document identity and replaces all embeddings atomically on
success. Previous embeddings remain on failure and stay searchable when compatible
with current application settings. A configuration change does not automatically
process documents or change their status. Search excludes incompatible model,
dimension, provider, or preprocessing snapshots; chunking changes alone do not hide
old results. Concurrent attempts for the same document fail with a conflict. After
a crashed process releases its database connection, a new invocation can retry a
stranded `processing` record. Application settings changed during processing require
a retry. Normal retries work from any status.

Run `rypiq db:migrate` before using these commands against an existing database.
It marks existing documents with embeddings as `done`, and empty parents as `pending`.
The configured Python API exposes `process_document(document_id=...)` and
`enumerate_documents(application_id=..., status='error')`. Injected `CreateDocument`
creates a pending stored original; `ProcessDocument` handles processing separately
for future service or worker integrations. No worker is started automatically.

Missing show targets exit nonzero. Delete returns `{"deleted": true}` or
`{"deleted": false}` for an absent target. Deleting the final embedding leaves an
empty parent document. These UUID-based operations are not authentication boundaries.
There is no embedding update CLI. The former `embeddings:create`, `embed`, and
`ingest` commands have been removed.

```sh
rypiq documents:create --name "Example document" report.pdf --application-id APPLICATION_UUID --output document.json
rypiq documents:show --id DOCUMENT_UUID
rypiq embeddings:show --id EMBEDDING_UUID --output chunk.json --embeddings-full
rypiq embeddings:delete --id EMBEDDING_UUID
rypiq documents:delete --id DOCUMENT_UUID
```

### Document ingestion and search

Use `documents:create` above to ingest documents, then search within their application:

```sh
rypiq documents:create --name "Example document" samples/sample.txt --application-id APPLICATION_UUID
rypiq search 'What is this document about?' --application-id APPLICATION_UUID --limit 5
```

Search returns a JSON array of chunk matches. Lower cosine distances are closer.
Search does not require object storage.

| Search option | Default or requirement |
| --- | --- |
| `--application-id UUID` | Required; selects the application and its embedding model. |
| `--limit N` | `5`; positive maximum number of matches. |
| `--metadata-filter JSON` | Optional JSON object matched against chunk metadata using containment. |
| `--model MODEL` | Unset: use the application's model. Local path or catalog alias override; the resolved filename must match the saved model. |
| `--context_size N` | Saved application/catalog setting, otherwise `2048`; positive local runtime override. |
| `--gpu_layers N` | Saved application/catalog setting, otherwise `0`; local runtime override, integer at least `-1`. |

Model and runtime overrides are rejected for remote applications. These overrides
apply to search, not to `documents:create`.

Search returns individual chunk matches with `embedding_id`, `document_id`,
`content`, `metadata`, parent `storage_key`, and cosine `distance`. Add
`--metadata-filter '{"start": 0}'` to restrict matches to chunks whose JSONB metadata contains that object.
Filters use JSONB containment before ranking and limiting. Search remains scoped
to the required application ID and does not perform keyword search.

### Inference commands

`rypiq chat PROMPT --application-id UUID` performs non-streaming retrieval-augmented inference. The prompt is positional and `--application-id` is required; there are no model or generation flags.

Chat embeds the question using the application's embedding settings, retrieves its top five matching chunks, and supplies them to its configured inference model. An application must have an inference model configured. The grounding prompt instructs the model to acknowledge insufficient evidence instead of answering from general knowledge, including when no chunks are retrieved. This is an instruction to the model, not a guarantee of factual correctness.

Chat returns the existing JSON fields `text`, `model`, `finish_reason`, and `usage`, plus `sources`. Each source contains `document_id` (a string UUID), `name`, `filename`, `source`, and `indicator` (such as `[1]`). Documents are deduplicated by ID in first-match retrieval order; no matches returns `sources: []`. Empty filenames and source values are preserved. The system prompt requests Markdown in `text` and inline indicators after quotations and sourced claims, for example `The guide recommends this. [1]`. Indicators start at `[1]` for each response and match the structured sources; chunks from the same document share an indicator. These are retrieved documents, not a guarantee that the model used each one. Citation accuracy and Markdown formatting depend on the model; responses are not post-validated. Passage text and embedding vectors are not included. It does not persist conversation history. CLI generation uses a maximum of 512 output tokens and the provider's default temperature. Python callers can supply conversation history and generation options.

```sh
rypiq chat 'What does the guide say about setup?' --application-id APPLICATION_UUID
```

The system prompt is `RAG_SYSTEM_PROMPT` in [inference/prompts.py](app/operations/inference/prompts.py). See [RAG prompt customization](docs/PROVIDERS.md#rag-prompt-customization) for the reusable operation and prompt-builder interface.

For reuse across Python calls, see [local model sessions](docs/PROVIDERS.md#reusing-local-models). Separate CLI invocations do not share loaded models.

### Model discovery

`rypiq models:list` lists configured catalog entries, not every model file on disk. Local entries report file availability; listing does not load model weights.

| Option | Default | Meaning |
| --- | --- | --- |
| `--application-id UUID` | Unset | Filter catalog entries to the application's provider; requires a database lookup. |
| `--discover` | Disabled | Requires `--application-id`; query configured remote endpoints and merge discovered models with catalog entries. For local applications, only the filtered catalog is returned. |

Without either option, listing reads only the catalog and needs no database. Discovery requires the remote endpoint and credential configuration; see the [model provider guide](docs/PROVIDERS.md).

```sh
rypiq models:list
rypiq models:list --application-id APPLICATION_UUID
# For an application configured with OpenAI or llama-server:
rypiq models:list --application-id APPLICATION_UUID --discover
```

### Object storage

| Syntax | Positional argument | Behavior |
| --- | --- | --- |
| `rypiq storage:save PATH` | Local file path | Store an original file without embedding or indexing it. |
| `rypiq storage:list` | None | List stored objects. |

| Option | Applies to | Default | Meaning |
| --- | --- | --- | --- |
| `--prefix PREFIX` | Save | `documents` | Logical prefix for the new object key. |
| `--prefix PREFIX` | List | Empty string | Filter object keys by prefix; empty lists all objects. |

Commands use `config/storage.yml` (or `RYPIQ_STORAGE_CONFIG`) and return object metadata: key, size, content type, and checksum. They do not require an application or database connection.

```sh
rypiq storage:save samples/sample.txt --prefix documents
rypiq storage:list --prefix documents
```

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

## HTTP API and document worker

RyPiq also provides a FastAPI service using the same operations and PostgreSQL backend.
See [API endpoints and examples](docs/API.md) for the complete contract.

Install dependencies and set `APP_SECRET_KEY` in `.env` to a generated secret:

```sh
env/bin/python -m pip install -e ".[dev]"
env/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Initialize a new database with `rypiq db:create`, or run `rypiq db:migrate` on an
existing database to add the durable processing queue. The server does not migrate
the database automatically.

Run these in separate terminals from the project directory:

```sh
env/bin/rypiq server --host 127.0.0.1 --port 3000 --reload
```

```sh
env/bin/rypiq worker
```

Both processes use the same database, model configuration, and object-storage root.
Open <http://127.0.0.1:3000/docs> and use **Authorize** to enter the secret. Every
`/api/v1` endpoint requires `X-App-Secret-Key`; `/docs` and `/openapi.json` are public
schema documentation. No user accounts or JWTs are involved.

CORS allows all browser origins by default. Set `API_CORS_ORIGINS` to a
comma-separated list such as `http://localhost:5173,https://frontend.example`
to restrict origins, or an empty value to disable cross-origin access. Browser
preflight requests are supported; actual API calls still require the secret header.

Document upload stores the original and returns `pending`. Submit the separate
processing endpoint to enqueue embeddings, then poll its returned URL. Accepted
work survives API and worker restarts in PostgreSQL. Chat, search, and the other
HTTP operations are synchronous. Existing CLI document creation remains synchronous.

## Injectable configuration

Use `app.config.load_config()` to capture settings once, then pass `config=` to
`create_app`, configured Python operations, context helpers, or worker functions.
Tests and embedding applications can construct `AppConfig` directly without
loading environment variables or YAML. See [configuration injection](docs/CONFIGURATION.md)
for typed settings, overrides, validation, and examples. Running servers and workers
pick up file/environment changes after restart; application records remain live.
