# Model providers

RyPiq selects embedding and inference adapters from an application's `provision_type`: `local`, `llama-server`, or `openai`. Both capabilities use that selection, but their models, runtime settings, and remote endpoints are independent. The [database guide](DATABASE.md) describes application persistence and CRUD; the [README](../README.md) covers installation.

## Common contracts and resource ownership

[Provider contracts](../app/providers/contracts.py) expose:

- `EmbeddingProvider.embed(texts) -> list[list[float]]`, `model_name`, and `close()`.
- `InferenceProvider.chat(messages, max_tokens=512, temperature=None) -> ChatResult` and `close()`.
- `ModelDiscovery.list_models() -> list[ModelInfo]` and `close()`.

Chat messages contain a text `content` and a `system`, `user`, or `assistant` role. `ChatResult` contains `text`, the reported `model`, `finish_reason`, and optional token `usage`. This is non-streaming plain inference: it does not retrieve documents, store chat history, call tools, or accept images/audio.

Operations borrow providers and never close them. Configured entrypoints own and close their clients/models, including failure paths, unless a caller supplies a local model session. Only the requested capability is constructed: ingestion/search does not load the inference model. Local inference uses a separate `llama_cpp.Llama` instance from embeddings.

Remote adapters use the OpenAI Python SDK with automatic retries disabled and a default 60-second request timeout. Their exceptions omit raw server diagnostics, credentials, and prompts. The llama-server adapter uses the server's OpenAI-compatible endpoints; it neither launches a server nor uses a local model instance for HTTP calls. See the [llama-cpp-python HTTP client example](https://llama-cpp-python.readthedocs.io/en/latest/server/).

## Reusing local models

Use `LocalModelSession` to share loaded models across Python operations. Loading is lazy: creating a session loads nothing, and embedding alone does not load inference. Optional `warmup()` accepts resolved `ProviderSettings` for whichever capabilities you want ready before requests.

```python
from uuid import UUID

from app.application import show_application, ingest_file, search_documents, chat
from app.providers.config import settings_for
from app.providers.contracts import ChatMessage
from app.providers.session import LocalModelSession

application_id = UUID('00000000-0000-0000-0000-000000000001')  # Replace with your local application ID.
application = show_application(application_id=application_id)

with LocalModelSession() as models:
    # Optional; omit inference warmup if the application has no inference model.
    models.warmup(settings_for(application, 'embedding'), settings_for(application, 'inference'))
    documents = ingest_file(path='samples/sample.txt', chunk_size=500, stride=400,
                            application_id=application_id, model_session=models)
    matches = search_documents(query='What is this about?', application_id=application_id,
                               model_session=models)
    answer = chat(messages=[ChatMessage('user', 'Say hello.')], application_id=application_id,
                  model_session=models)
```

`embed_file`, `embed_text_file`, and the legacy `ingest_document` helper also accept `model_session`. Without one, each call continues to load and close its own provider. Remote providers ignore the supplied local session and retain their existing lifecycle. No database or storage resources move into the session.

A session retains one successful load per capability, canonical absolute model path (resolving symlinks), context size, GPU layer count, and inference chat format. Matching settings across applications reuse the same instance. Embedding and inference remain separate instances, even for the same file. Prefixes, embedding-size validation, temperature, and token limits remain operation-specific. Settings for `warmup()` must already have aliases resolved; configured entrypoints resolve aliases before borrowing models.

Concurrent first use constructs a matching provider once; each provider serializes its own model execution. Failed loads can be retried. Successfully loaded providers remain cached after an operation or later warmup fails. Session shutdown rejects new work, waits for active leases, and attempts every model's cleanup once. Repeated `close()` calls are safe, and closed sessions cannot be reopened. Cleanup errors preserve a primary operation exception, or raise a safe `ProviderError` if no other exception is active.

Advanced callers can borrow a provider using `with models.open_provider(settings) as provider:` and inject it into an operation. Do not close borrowed providers or retain them beyond that block. Close the session outside all leases; closing while the same thread holds a lease raises an error to avoid deadlock. Finish or join worker tasks before leaving the session context. Sessions are for threads within one process, not for sharing across processes.

Custom factories belong on `LocalModelSession(embedding_factory=..., inference_factory=...)`; combining a session with a per-call embedding factory raises `ValueError`. Models remain resident until shutdown, so distinct runtime configurations consume additional memory. There is no automatic eviction or model-file hot reload: start a new session after replacing model files. Separate sessions and separate CLI invocations do not share loaded models.

## Repository model catalog

[config/models.yml](../config/models.yml) lists configured models. Set `RYPIQ_MODELS_CONFIG` to choose another YAML file, including outside the checkout. Catalog paths resolve relative to that file's directory, with `~` expansion supported.

```yaml
models:
  embedding:
    provider: local
    capabilities: [embedding]
    path: ../models/embeddinggemma-300m-Q4_0.gguf
    options:
      document_prefix: 'title: none | text: '
      query_prefix: 'task: search result | query: '
  chat:
    provider: local
    capabilities: [inference]
    path: ../models/your-chat-model.gguf
    options:
      context_size: 4096
      gpu_layers: 0
  remote-embedding:
    provider: openai
    capabilities: [embedding]
    model: text-embedding-3-small
    options:
      dimensions: 768
```

Reference an alias with `@`, for example `--embedding-model @embedding`. Unprefixed values are direct local paths or remote model IDs. Aliases must exist and declare the selected provider and capability. Models are never downloaded automatically. The shipped `@local-chat` entry is a placeholder path; replace it with your own inference model.

Configured application creation/update resolves aliases immediately and persists the resolved model and catalog defaults. Later catalog edits do not alter saved applications. Lower-level CRUD services accept already resolved dataclasses; callers using them directly should resolve aliases explicitly with `resolve_application(application, load_catalog())` at their composition boundary.

Configuration precedence is: explicit allowed CLI override, saved application options, catalog defaults copied when saving, then provider defaults. The standalone Python `embed_file()` API and `chat` CLI resolve catalog aliases on each call. Ad-hoc paths remain supported. New applications created through the CLI or configured Python entrypoint store direct local model paths as absolute paths resolved from the creation directory (including `~` and symlinks). Existing application records and direct CRUD service inputs retain their stored paths; use absolute paths when updating models. Local model identity uses the resolved filename, including for symlink overrides.

## Application provider configuration

For a complete local setup including PostgreSQL, object storage, and a model catalog, see the [configuration examples](../README.md#configuration-examples).

Store options under `Application.data.provider_config`, with `embedding` and `inference` sections. `applications:create --data` and `applications:update --data` accept the entire JSON data object; updates replace that object rather than patching nested keys. Keep unrelated metadata when replacing it.

Application chunking is stored separately under `data.chunking` (`chunk_size` and `stride`), defaulting to 500 and 400. Use `applications:create` or `applications:update` with `--chunk-size` and `--stride` to set these values; explicit flags override corresponding values in `--data`. Chunking changes are allowed while documents exist and only affect future uploads.

Common options are `base_url`, `api_key_env`, and `timeout`. Local runtime options are `context_size` (default 2048) and `gpu_layers` (default 0; -1 is accepted for all layers). Local inference also accepts `chat_format`; omit it to use llama-cpp-python's model template selection. Local runtime options have no effect on remote server execution; configure those servers separately.

Embedding options are `document_prefix`, `query_prefix`, and `dimensions`. The dimensions request is omitted unless configured; if supplied it must equal `Application.embedding_size`. The selected upstream model must support it. Regardless of requested dimensions, every returned vector is validated against the application's size. See [OpenAI embedding parameters](https://developers.openai.com/api/reference/resources/embeddings/methods/create).

Local applications without explicit prefixes retain the existing EmbeddingGemma document/query prefixes. Remote applications default to empty prefixes. Configure the appropriate prefixes for other local models, including empty strings where none are required. These formatting choices are model-specific, not transport-specific. Existing documents are never silently re-embedded.

Application updates cannot change embedding prefixes, requested dimensions, model, provision type, or embedding size while documents exist. Inference settings, endpoint locations, and credential references remain editable. An endpoint change must still serve the same compatible model.

### Local

```sh
rypiq applications:create --name 'Local example' \
  --embedding-model @embeddinggemma --embedding-size 768 \
  --inference-model @local-chat
```

Local models must be existing GGUF files. `--model` on application-scoped search overrides the path only when its filename matches the selected local model; it cannot switch the application's embedding model. `--context_size` and `--gpu_layers` override local search runtime settings only when explicitly supplied. `embeddings:create` uses saved application settings and accepts none of these overrides.

### Existing llama.cpp server

```sh
rypiq applications:create --name 'Server example' --provision-type llama-server \
  --embedding-model embedding-alias --embedding-size 768 --inference-model chat-alias \
  --data '{"provider_config":{"embedding":{"base_url":"http://localhost:8081/v1"},"inference":{"base_url":"http://localhost:8082/v1"}}}'
```

Use the model IDs/aliases served by your endpoints, including `/v1` in each base URL. Embedding and chat may use the same URL or separate servers. RyPiq calls `/embeddings`, `/chat/completions`, and `/models` relative to these base URLs. Servers must support the requested endpoints and embedding pooling/output shape.

Authentication is optional. If required, set `api_key_env` in the appropriate section to an environment-variable name such as `LLAMA_SERVER_API_KEY`, then export that variable. No actual credential belongs in application JSON or the catalog. URLs cannot contain embedded credentials, query parameters, or fragments.

### OpenAI

```sh
export OPENAI_API_KEY='your-provider-key'
rypiq applications:create --name 'OpenAI example' --provision-type openai \
  --embedding-model @openai-embedding --embedding-size 768 \
  --inference-model YOUR_CHAT_COMPLETIONS_MODEL_ID
```

The default base URL is `https://api.openai.com/v1`; the default credential reference is `OPENAI_API_KEY`. Override them separately per capability if needed. `Application.api_key` is a RyPiq application credential and is never sent to upstream providers.

Model IDs are not hard-coded to a whitelist. Inference requires a model supporting the Chat Completions endpoint; models exclusive to other endpoints are outside this contract. `max_tokens` maps to OpenAI's `max_completion_tokens` and llama.cpp's `max_tokens`. Temperature is omitted unless supplied, because model support varies. See the [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

## Chat and discovery commands

```sh
rypiq chat 'Explain cosine similarity.' --application-id APPLICATION_UUID \
  --system 'Keep the explanation brief.' --max-tokens 256
rypiq chat 'Hello' --model @local-chat
rypiq chat 'Hello' --model /absolute/path/to/chat.gguf --temperature 0.5
rypiq models:list
rypiq models:list --application-id APPLICATION_UUID --discover
rypiq applications:update APPLICATION_UUID --inference-model @local-chat
rypiq applications:update APPLICATION_UUID --clear-inference-model
```

Chat requires exactly one application ID or standalone local model. `--max-tokens` defaults to 512. A missing application inference model fails before constructing a provider. Chat, model discovery, and application show/update return JSON. `applications:list` defaults to text; use `--output json` for JSON. Application list/show/update output omits the application's API key.

Without `--discover`, model listing is catalog-only and performs no remote calls. An application ID filters the catalog by provider. Discovery requires an application ID and queries each distinct configured remote endpoint/credential pair once; a separate inference endpoint can be listed before choosing its inference model. Local discovery only checks catalog file availability and does not load weights or scan directories.

Results are deterministic and deduplicated by provider and resolved model. They include alias/ID, resolved model, capabilities, source (`catalog`, `discovered`, or `catalog+discovered`), and availability (`available`, `missing`, or `unverified`). Remote discovered capabilities remain empty unless declared in the catalog; names are not used to guess capabilities. A catalog declaration is not proof that a remote model is currently accessible. Discovery never changes the catalog. See [OpenAI model listing](https://developers.openai.com/api/reference/resources/models/methods/list).

## Python usage

```python
from uuid import UUID
from app.application import chat, list_models
from app.providers.contracts import ChatMessage

result = chat(
    application_id=UUID('00000000-0000-0000-0000-000000000001'),
    messages=[ChatMessage('system', 'Be brief.'), ChatMessage('user', 'Hello')],
)
models = list_models()  # Catalog only.
```

For caller-owned resources, inject an inference provider into `app.operations.chat.Chat` and call `execute()`. Ingestion and search retain their existing injected operations and configured entrypoints. Existing result shapes and document embedding snapshots remain unchanged. `ingest_document()` remains the legacy local composition helper; use `IngestDocument` or configured `ingest_file()` for remote applications.

## Validation

Provider unit tests cover configuration, catalog resolution, local runtime interaction, normalization, and cleanup. Loopback HTTP tests exercise the actual OpenAI SDK without API credits:

```sh
env/bin/python -m pytest tests/unit/embeddings tests/unit/operations -q
env/bin/python -m pytest tests/integration/embeddings/test_http_providers.py -q
```

Database and existing local embedding integrations use the prerequisites described in [Database](DATABASE.md#database-integration-tests). Live remote services are not contacted by ordinary tests. Local inference validation needs an existing chat-capable model; set `INFERENCE_MODEL_PATH` for the opt-in test. No model is downloaded to run tests.

To explicitly run live remote checks, set `RYPIQ_LIVE_OPENAI_CONFIG` or `RYPIQ_LIVE_LLAMA_SERVER_CONFIG` to a JSON file containing `embedding_model`, `embedding_size`, `inference_model`, and `data.provider_config` as described above. Export the referenced credential variables separately. Then run `tests/integration/embeddings/test_live_remote_providers.py`. Setting these paths opts into outbound provider requests; OpenAI requests may incur charges. Without them, the live checks are skipped and only loopback HTTP compatibility is verified.
