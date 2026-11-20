# HTTP API

The service follows the [default_api_fast conventions](https://github.com/cloudband-solutions/default_api_fast):
app factory, thin resource controllers, Pydantic schemas, central route registration,
and independent operations. RyPiq retains its explicit PostgreSQL schema and
repository interfaces. It does not introduce SQLAlchemy ORM models, Alembic, user
accounts, or a second database backend.

## Running locally

Use the [normal setup](../README.md) for PostgreSQL with pgvector, storage, and models.
Install the updated package with `env/bin/python -m pip install -e ".[dev]"`.
Generate a secret with `env/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))'`
and put it in your untracked `.env`:

```dotenv
APP_SECRET_KEY=<your-generated-secret>
API_MAX_UPLOAD_BYTES=104857600
```

`APP_SECRET_KEY` is required and must not be blank. The API captures settings at
startup; restart it after configuration changes. Python callers can supply
`create_app(config=...)`; see [configuration injection](CONFIGURATION.md). Existing process environment
values take precedence over `.env`. It is independent of the existing application
record's `api_key`, which is neither accepted for HTTP access nor returned by the API.

For a new database:

```sh
env/bin/rypiq db:create
```

For an existing database:

```sh
env/bin/rypiq db:migrate
```

Start the API and worker in separate terminals:

```sh
env/bin/rypiq server --host 127.0.0.1 --port 3000 --reload
```

```sh
env/bin/rypiq worker
```

The equivalent ASGI command is:

```sh
env/bin/python -m uvicorn app:create_app --factory --host 127.0.0.1 --port 3000 --reload
```

`worker --once` claims at most one job, which is useful for controlled execution.
The normal worker polls every second and executes one job at a time. Additional
worker processes can share the queue; each needs access to the same storage root
and model files. No Redis, Celery, or separate queue service is required.

The default bind address is localhost. Reload is opt-in.
The API does not create databases or run migrations at startup.

## Browser access (CORS)

CORS allows all origins (`*`) by default. To restrict browser access, set a
comma-separated origin list in `.env` and restart the server:

```dotenv
API_CORS_ORIGINS=http://localhost:5173,https://frontend.example
```

Origins include the scheme and port, with no path or trailing slash. An empty
`API_CORS_ORIGINS` disables cross-origin access. Browser preflight `OPTIONS`
requests do not require the secret; actual API requests still require
`X-App-Secret-Key`. Allowed methods are GET, POST, PATCH, DELETE, and OPTIONS;
allowed request headers include Content-Type and X-App-Secret-Key. The Location
response header is exposed so browser clients can read the job polling URL.
Cookie credentials are disabled; use the secret header without
`credentials: "include"`. Allowed origins receive CORS headers on errors as well
as successful responses.

## Authentication and responses

All service requests under `/api/v1`, including health and doctor, require the
following header (browser CORS preflights are handled separately):

```http
X-App-Secret-Key: <your-generated-secret>
```

Credentials are compared in constant time. Invalid, missing, or duplicate headers
receive `401` before request bodies are parsed or business resources are opened.
The public `/docs` and `/openapi.json` routes expose schemas only; Swagger's
**Authorize** button sets the header for service calls.

Responses use domain objects or arrays, without a universal envelope. UUIDs are
strings and timestamps use ISO 8601. Application API keys, worker claim tokens,
and temporary upload paths are omitted. Errors use `detail` and never include raw
database/provider diagnostics. Request-validation errors omit submitted values.

| Status | Meaning |
| --- | --- |
| `200` | Successful read, update, processing, or deletion |
| `201` | Application, stored document, or storage object created |
| `202` | Document embedding job accepted, or an active job already exists |
| `401` | Missing or incorrect service secret |
| `404` | Required resource does not exist |
| `409` | Conflicting state, duplicate document name, or active work prevents deletion |
| `413` | Upload exceeds the configured limit |
| `422` | Invalid request or document content |
| `502` | Model provider failure |
| `503` | Infrastructure/configuration unavailable, or doctor finds required setup incomplete |
| `500` | Unexpected failure; safe diagnostic only |

Deletes preserve CLI behavior: `{ "deleted": false }` for an absent record, rather
than a `404`. Deleting documents or embeddings preserves object-storage files.
An upload failure after creation may return `document_id` so the persisted error
record can be inspected. File storage and database transactions are separate;
existing storage compensation and document failure-status rules still apply.

## Routes

Every path in this table is relative to `/api/v1`.

| Method | Path | Input / result |
| --- | --- | --- |
| GET | `/health` | `{ "status": "ok" }`; no dependency checks |
| GET | `/doctor` | Read-only configuration, database, storage, and model-catalog report |
| GET | `/applications` | Application summaries and document counts |
| POST | `/applications` | Application configuration; returns created record without API key |
| GET | `/applications/{id}` | Application configuration |
| PATCH | `/applications/{id}` | Only supplied fields change |
| DELETE | `/applications/{id}` | Delete application if it has no documents |
| GET | `/applications/{id}/documents` | Summaries; optional `status=pending\|processing\|error\|done` |
| POST | `/applications/{id}/documents` | Multipart `name` and `file`; store only |
| GET | `/documents/{id}` | Document and ordered chunk summaries |
| DELETE | `/documents/{id}` | Delete document/chunks; rejects active jobs |
| POST | `/documents/{id}/process` | Optional JSON `{ "batch_size": 1 }`; enqueue/reprocess |
| GET | `/documents/{id}/status` | Document status and latest job summary |
| GET | `/processing-jobs/{id}` | Specific job state, attempts, timestamps, and safe error |
| GET | `/embeddings/{id}` | Complete chunk including vector |
| DELETE | `/embeddings/{id}` | Delete a chunk |
| POST | `/applications/{id}/search` | JSON `query`, optional `limit` (5), `metadata_filter` object |
| POST | `/applications/{id}/chat` | JSON `prompt`; synchronous RAG answer and sources |
| GET | `/models` | Optional `application_id` and `discover=true` |
| POST | `/processing/text` | JSON `content`; text statistics |
| POST | `/processing/text-file` | Multipart `file`, `chunk_size`, `stride`; chunks with offsets |
| POST | `/storage` | Multipart `file`, optional `prefix` (`documents`) |
| GET | `/storage` | Optional `prefix`; stored object descriptions |

Application creation accepts `name`, `embedding_model`, optional `embedding_size`
(defaulting to database dimensions, normally 768), independent
`embedding_provision_type` and `inference_provision_type` (default `local`),
`inference_model`, `data`, `chunk_size`, and `stride`. These have the same meaning
as the CLI options. PATCH accepts the same fields; omitted fields are preserved,
`inference_model: null` clears that field, and `data` replaces the whole object.
Other explicitly null fields are rejected.

Uploads accept bytes rather than server filesystem paths. Client filenames are
reduced to their basename; extraction keeps the extension. The file-size limit
is 100 MiB by default, with an additional 1 MiB allowance for the complete request's
multipart overhead. Temporary files are removed on both success and failure.

CLI-only options such as output export paths, vector previews, and terminal tables
have no HTTP equivalent. Database create/migrate/drop/reset remain CLI-only.

## Example document workflow

In a shell, export the same secret configured for the API. `.env` loading by the
server does not automatically export variables into your shell.

```sh
curl http://127.0.0.1:3000/api/v1/doctor \
  -H "X-App-Secret-Key: $APP_SECRET_KEY"

curl -X POST http://127.0.0.1:3000/api/v1/applications \
  -H "X-App-Secret-Key: $APP_SECRET_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"name":"Local documents","embedding_model":"/absolute/path/to/model.gguf"}'
```

Set `APPLICATION_ID` to the returned ID, then upload:

```sh
curl -X POST "http://127.0.0.1:3000/api/v1/applications/$APPLICATION_ID/documents" \
  -H "X-App-Secret-Key: $APP_SECRET_KEY" \
  -F 'name=Example report' -F 'file=@report.txt'
```

Set `DOCUMENT_ID` to the returned document ID. Upload returns `pending` and does
not enqueue or load a model. Start embedding explicitly:

```sh
curl -X POST "http://127.0.0.1:3000/api/v1/documents/$DOCUMENT_ID/process" \
  -H "X-App-Secret-Key: $APP_SECRET_KEY" \
  -H 'Content-Type: application/json' -d '{"batch_size":1}'

curl "http://127.0.0.1:3000/api/v1/documents/$DOCUMENT_ID/status" \
  -H "X-App-Secret-Key: $APP_SECRET_KEY"
```

Submission returns `job_id`, `document_id`, `state`, and `polling_url`, with the
same URL in the `Location` response header. Poll that URL to track this exact
attempt even if a later reprocessing job is submitted.

```sh
curl -X POST "http://127.0.0.1:3000/api/v1/applications/$APPLICATION_ID/search" \
  -H "X-App-Secret-Key: $APP_SECRET_KEY" \
  -H 'Content-Type: application/json' -d '{"query":"What does the report say?","limit":5}'
```

## Queue and recovery guarantees

Jobs are committed to PostgreSQL before `202` is returned. They remain queued
when no worker is running. A partial unique index permits one queued/running job
per document. Duplicate submissions return that existing job, retaining its
original `batch_size`; after a terminal job, resubmission creates a new attempt.

Job states are `queued`, `running`, `succeeded`, and `failed`. Document states remain
`pending`, `processing`, `done`, and `error`. A document may still be `done` while
its replacement job is queued; inspect the job state as well as document status.
Processing uses the application configuration available when the worker starts.
Changes during processing are checked before committing replacement embeddings.

Workers claim using short transactions and `FOR UPDATE SKIP LOCKED`. Claims expire
after 60 seconds, with heartbeats every 10 seconds. A claim token fences document
writes. Successful embedding replacement, document completion, and job completion
commit together. Existing document advisory locks also prevent overlapping CLI
and worker processing. A busy lock releases the claim without consuming an attempt and moves it behind
other waiting work.

Interrupted claims are recovered up to three claimed attempts. Ordinary processing
errors fail immediately and require explicit resubmission. Expired workers cannot
commit results or overwrite a newer attempt's status. Previous embeddings and the
stored original survive processing failures. Execution is **at least once**:
provider calls can repeat after a crash even though database publication is fenced.

Graceful shutdown stops new claims and finishes the current attempt. Forced
termination is recovered after lease expiry. There is no cancellation endpoint or
per-chunk progress percentage. Terminal job history remains until its document is
deleted; active jobs block document deletion through both API and CLI. Operators
can inspect `GET /doctor`, job polling responses, and safe worker lifecycle logs.
