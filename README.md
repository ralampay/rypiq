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

Library callers should use `with create_context() as context:` from `app.context` or `with get_database() as database:` from `app.database.factory` to release connection pools reliably.

## CLI reference

Commands return JSON; failures exit nonzero with a diagnostic. Use `rypiq --help` to list commands and `rypiq COMMAND --help` for individual usage. `-h` is also supported. Run the examples from the project root and replace `APPLICATION_UUID` with an existing application's UUID. Flag spelling matters: chunking and local runtime flags use underscores, while application and chat flags use hyphens.

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
| `rypiq application:create` | `--name NAME --embedding-model MODEL` | Create an application and generate its API key. |
| `rypiq application:list` | None | List applications with pagination. |
| `rypiq ls-applications` | None | List every application's ID and name; `--output text` (default) or `--output json`. |
| `rypiq application:show APPLICATION_UUID` | Application UUID positional argument | Show one application's details. |
| `rypiq application:update APPLICATION_UUID` | Application UUID positional argument | Update supplied fields; omitted fields remain unchanged. |

Creation returns the generated API key. List, show, and update omit it. There is no application deletion CLI command.

`ls-applications` lists all applications in UUID order, without pagination arguments.
The default text table has `ID` and `NAME` columns; control characters in names are
escaped so each application occupies one row. An empty database prints only the
headers. JSON output preserves names and contains only `id` and `name`, returning
`[]` when empty. Listing does not guarantee a snapshot during concurrent changes.

```sh
rypiq ls-applications
rypiq ls-applications --output text
rypiq ls-applications --output json
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
| `--embedding-size N` | Database configuration's `dimensions` (normally `768`); range `1–16000` | Replace the embedding size. |
| `--provision-type TYPE` | `local`; choices: `local`, `openai`, `llama-server` | Replace the provider. |
| `--inference-model MODEL` | Unset | Set the inference model. |
| `--data JSON` | `{}`; must be a JSON object | Replace the entire data object, not a partial merge. |
| `--clear-inference-model` | Not available on create | Clear the inference model; mutually exclusive with `--inference-model`. |

Embedding model, size, provider, and embedding preprocessing changes are rejected when the application already has documents. See [database documentation](docs/DATABASE.md) for persistence rules.

| List option | Default | Meaning |
| --- | --- | --- |
| `--limit N` | `100` | Maximum applications returned; positive integer. |
| `--offset N` | `0` | Applications to skip; nonnegative integer. |

```sh
rypiq application:create --name 'My application' \
  --embedding-model models/embeddinggemma-300m-Q4_0.gguf --embedding-size 768 \
  --inference-model models/your-chat-model.gguf --data '{"purpose":"local search"}'
rypiq application:list --limit 20 --offset 0
rypiq application:show APPLICATION_UUID
rypiq application:update APPLICATION_UUID --name 'Renamed application'
rypiq application:update APPLICATION_UUID --clear-inference-model
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
rypiq embed text-file presentation.pptx --chunk_size 500 --stride 400
rypiq ingest workbook.xlsx --application-id APPLICATION_UUID --chunk_size 500 --stride 400
```

### Embedding commands

`rypiq embed text-file PATH --chunk_size N --stride N` embeds a supported document with a local model. It requires neither an application nor a database and does not store documents.

| Option | Default or requirement | Meaning |
| --- | --- | --- |
| `--chunk_size N` | Required | Maximum characters per chunk; positive integer. |
| `--stride N` | Required | Positive character advance, no greater than chunk size. |
| `--model MODEL` | `models/embeddinggemma-300m-Q4_0.gguf` | Local GGUF path or embedding-capable local `@catalog-alias`. |
| `--context_size N` | Catalog setting, otherwise `2048` | Positive local model context size. |
| `--gpu_layers N` | Catalog setting, otherwise `0` | Local GPU layer count; `-1` requests all layers. |
| `--batch_size N` | `1` | Positive number of chunks per embedding request. |
| `--output PATH` | Unset: print JSON to stdout | Write the complete JSON result to a file instead; replaces an existing file. |

```sh
rypiq embed text-file samples/sample.txt --chunk_size 500 --stride 400 \
  --batch_size 2 --output embeddings.json
```

### Document ingestion and search

| Syntax | Positional argument | Behavior |
| --- | --- | --- |
| `rypiq ingest PATH --application-id APPLICATION_UUID --chunk_size N --stride N` | Supported document path | Save the original file, embed its chunks, and persist application-owned documents. |
| `rypiq search QUERY --application-id APPLICATION_UUID` | Search text, quoted if it contains spaces | Return matching documents from that application with cosine distances; lower is closer. |

Both commands require the database and use the application's configured provider (`local`, `openai`, or `llama-server`). Only ingestion requires object storage.

| Option | Applies to | Default or requirement |
| --- | --- | --- |
| `--application-id UUID` | Both | Required; selects the application and its models. |
| `--model MODEL` | Both | Unset: use the application's model. Local path or local catalog alias override only; the resolved filename must match the application's model. |
| `--context_size N` | Both | Saved application/catalog setting, otherwise `2048`; positive local runtime override. |
| `--gpu_layers N` | Both | Saved application/catalog setting, otherwise `0`; local runtime override, integer at least `-1`. |
| `--chunk_size N` | Ingest | Required; positive maximum characters per chunk. |
| `--stride N` | Ingest | Required; positive character advance no greater than chunk size. |
| `--batch_size N` | Ingest | `1`; positive number of chunks per embedding request. |
| `--limit N` | Search | `5`; positive maximum number of matches. |

Explicit `--model`, `--context_size`, and `--gpu_layers` overrides are rejected for remote applications. Ingestion returns `indexed_chunks` and `storage_key`; search returns a JSON array with document IDs, distances, content, and storage keys.

```sh
rypiq ingest samples/sample.txt --application-id APPLICATION_UUID --chunk_size 500 --stride 400
rypiq search 'What is this document about?' --application-id APPLICATION_UUID --limit 5
```

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
