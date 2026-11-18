# RyPiq

A Python CLI and library for processing text, generating embeddings and text responses, storing original files, and searching documents with PostgreSQL and pgvector.

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

## Model providers

See [Model providers](docs/PROVIDERS.md) for local GGUF execution, existing llama.cpp HTTP servers, OpenAI, the model catalog, chat, and model discovery.

## Ingest and search

Provide a compatible local GGUF embedding model; these commands do not download one.

```sh
rypiq applications:create --name 'My application' \
  --embedding-model models/embeddinggemma-300m-Q4_0.gguf --embedding-size 768
rypiq applications:list
# Use the UUID returned by applications:create:
rypiq embeddings:create samples/sample.txt --application-id APPLICATION_UUID
rypiq search 'What is this document about?' --application-id APPLICATION_UUID \
  --model models/embeddinggemma-300m-Q4_0.gguf
```

See [application commands](docs/DATABASE.md#application-commands) for configuration options and supported providers.

`embeddings:create` uses the application's saved model and chunking settings. Search also uses the application model and supports a compatible local `--model` path override. The former `embed` and `ingest` CLI commands have been removed; standalone embedding remains available through the Python API.

`config/storage.yml` controls original-file storage. `STORAGE_ENV` defaults to `DATABASE_ENV`; `RYPIQ_STORAGE_CONFIG` overrides its configuration path. Storage paths remain independent of database reset.

Library callers should use `with create_context() as context:` from `app.context` or `with get_database() as database:` from `app.database.factory` to release connection pools reliably.

## CLI reference

Commands return JSON except `applications:list`, which defaults to a text table and supports `--output json`. Failures exit nonzero with a diagnostic. Use `rypiq --help` to list commands and `rypiq COMMAND --help` for individual usage. `-h` is also supported. Run the examples from the project root and replace `APPLICATION_UUID` with an existing application's UUID. Flag spelling matters: chunking and local runtime flags use underscores, while application and chat flags use hyphens.

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

An application selects a provider and embedding/inference models. Create and update accept local GGUF paths, remote model IDs, or compatible `@catalog-alias` references. See [provider configuration](docs/PROVIDERS.md#application-provider-configuration) for endpoints and credential environment references.

| Command | Required arguments | Purpose |
| --- | --- | --- |
| `rypiq applications:create` | `--name NAME --embedding-model MODEL` | Create an application and generate its API key. |
| `rypiq applications:list` | None | List every application's ID and name; `--output text` (default) or `--output json`. |
| `rypiq applications:show APPLICATION_UUID` | Application UUID positional argument | Show one application's details. |
| `rypiq applications:delete APPLICATION_UUID` | Application UUID positional argument | Delete an application that has no documents. |
| `rypiq applications:update APPLICATION_UUID` | Application UUID positional argument | Update supplied fields; omitted fields remain unchanged. |

Creation returns the generated API key. List, show, and update omit it.

`applications:delete APPLICATION_UUID` returns `{"deleted": true}` when deleted or `{"deleted": false}` when already absent. Both deletion outcomes exit successfully. Applications with documents cannot be deleted and return a nonzero exit; stored files are never removed by this command. Deletion runs without an interactive prompt.

Application CLI commands use the plural `applications:*` domain. The former
`application:*` names and `ls-applications` are no longer supported.

`applications:list` lists all applications in UUID order, without pagination arguments.
The default text table has `ID` and `NAME` columns; control characters in names are
escaped so each application occupies one row. An empty database prints only the
headers. JSON output preserves names and contains only `id` and `name`, returning
`[]` when empty. Listing does not guarantee a snapshot during concurrent changes.

```sh
rypiq applications:list
rypiq applications:list --output text
rypiq applications:list --output json
```

Example JSON output:

```json
[
  {"id": "00000000-0000-0000-0000-000000000001", "name": "My application"}
]
```

| Create/update option | Creation default or requirement | Update behavior |
| --- | --- | --- |
| `--name NAME` | Required | Replace the name. |
| `--embedding-model MODEL` | Required | Replace the embedding model. |
| `--chunk-size N` | `500` when absent from application data | Replace saved maximum characters per chunk. |
| `--stride N` | `400` when absent from application data | Replace saved character advance; must be positive and no greater than chunk size. |
| `--embedding-size N` | Database configuration's `dimensions` (normally `768`); range `1–16000` | Replace the embedding size. |
| `--provision-type TYPE` | `local`; choices: `local`, `openai`, `llama-server` | Replace the provider. |
| `--inference-model MODEL` | Unset | Set the inference model. |
| `--data JSON` | `{}`; must be a JSON object | Replace the entire data object, not a partial merge. |
| `--clear-inference-model` | Not available on create | Clear the inference model; mutually exclusive with `--inference-model`. |

`--chunk-size` and `--stride` are saved as `data.chunking.chunk_size` and `data.chunking.stride`. Explicit flags override those values in `--data`; without `--data`, updates preserve other application data. Missing settings default independently to 500 and 400. Chunking changes affect future uploads and are allowed when documents already exist.

Embedding model, size, provider, and embedding preprocessing changes are rejected when the application already has documents. See [database documentation](docs/DATABASE.md) for persistence rules.

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
rypiq embeddings:create presentation.pptx --application-id APPLICATION_UUID
rypiq embeddings:create workbook.xlsx --application-id APPLICATION_UUID
```

### Embedding commands

| Command | Behavior |
| --- | --- |
| `rypiq embeddings:create PATH --application-id UUID` | Save the original file and persist an embedding record for each nonempty text chunk. |
| `rypiq embeddings:show --id UUID` | Show one persisted chunk, including text, vector, application ID, and metadata. |
| `rypiq embeddings:delete --id UUID` | Delete one chunk record; preserve other chunks and the stored original file. |

Creation uses the application's saved model, provider, dimensions, preprocessing,
and chunking settings. Configure chunking with `applications:create` or
`applications:update --chunk-size N --stride N`; missing settings default to
500 characters per chunk and a stride of 400. Creation does not accept model,
runtime, or chunking overrides. It requires the configured database and object
storage. Show and delete require only the database and a chunk UUID, with no
`--application-id` or model loading.

| Option | Applies to | Default or behavior |
| --- | --- | --- |
| `--application-id UUID` | Create | Required application configuration. |
| `--batch_size N` | Create | `1`; positive number of chunks per embedding request. |
| `--id UUID` | Show, delete | Required UUID of one chunk record, not its shared `source_id`. |
| `--output PATH` | Create, show | Optional full JSON export in addition to console output. No file is written by default. |
| `--embeddings-full` | Create, show | Print every vector value; otherwise vectors longer than six values show the first three, `"..."`, and last three. |

Creation prints `indexed_chunks`, `storage_key`, and `chunks`. Each chunk contains
its `id`, `application_id`, text in `content`, numeric `embedding`, model information,
source/storage details, metadata, and timestamps. Show prints one such record.
Exports always contain full numeric vectors, regardless of console truncation.
UUIDs are strings and timestamps use ISO format.

The export parent directory must already exist. An existing export file is
replaced atomically; the creation export cannot be the input document. Persistence
finishes before export. If export fails, the command exits nonzero and reports
that records were persisted, including chunk IDs for retrying with
`embeddings:show --id UUID --output PATH`. Stored records and originals are retained.

Show fails with a nonzero exit if the chunk does not exist. Delete returns
`{"deleted": true}` or `{"deleted": false}` if already absent; both exit successfully.
There is no `embeddings:update` command. Repeated creation creates new records.
The old `embed` and `ingest` CLI names are no longer accepted. Python callers can
still use `embed_file()` and `ingest_file()` with their existing arguments.

```sh
rypiq applications:update APPLICATION_UUID --chunk-size 500 --stride 400
rypiq embeddings:create samples/sample.txt --application-id APPLICATION_UUID
rypiq embeddings:create report.pdf --application-id APPLICATION_UUID --output embeddings.json
rypiq embeddings:show --id CHUNK_UUID --output chunk.json --embeddings-full
rypiq embeddings:delete --id CHUNK_UUID
```

### Document ingestion and search

Use `embeddings:create` above to ingest documents, then search within their application:

```sh
rypiq embeddings:create samples/sample.txt --application-id APPLICATION_UUID
rypiq search 'What is this document about?' --application-id APPLICATION_UUID --limit 5
```

Search returns a JSON array with chunk document IDs, cosine distances, content,
and storage keys. Lower distances are closer. Search does not require object storage.

| Search option | Default or requirement |
| --- | --- |
| `--application-id UUID` | Required; selects the application and its embedding model. |
| `--limit N` | `5`; positive maximum number of matches. |
| `--model MODEL` | Unset: use the application's model. Local path or catalog alias override; the resolved filename must match the saved model. |
| `--context_size N` | Saved application/catalog setting, otherwise `2048`; positive local runtime override. |
| `--gpu_layers N` | Saved application/catalog setting, otherwise `0`; local runtime override, integer at least `-1`. |

Model and runtime overrides are rejected for remote applications. These overrides
apply to search, not to `embeddings:create`.

### Inference commands

`rypiq chat PROMPT` performs non-streaming text inference. The prompt is a positional argument. Choose exactly one of `--application-id` and `--model`.

| Option | Default or requirement | Meaning |
| --- | --- | --- |
| `--application-id UUID` | Required unless `--model` is supplied | Use the application's configured inference model and provider. |
| `--model MODEL` | Required unless `--application-id` is supplied | Standalone local GGUF path or inference-capable local `@catalog-alias`; no database required. |
| `--system TEXT` | Unset | Prepend a system message. |
| `--max-tokens N` | `512` | Positive maximum output token count. |
| `--temperature N` | Unset: use the provider's default | Sampling temperature, finite number from `0` to `2`. |

An application must have an inference model configured to use `--application-id`. Chat returns text, model, finish reason, and optional usage; it does not retrieve documents or maintain conversation history.

```sh
rypiq chat 'Say hello.' --application-id APPLICATION_UUID --system 'Be concise.' --max-tokens 128
rypiq chat 'Say hello.' --model models/your-chat-model.gguf --temperature 0.5
```

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
