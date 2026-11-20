# Model providers

RyPiq selects adapters independently from an application's `embedding_provision_type` and `inference_provision_type`. Each defaults to `local` and accepts `local`, `llama-server`, or `openai`. Models, runtime settings, endpoints, and credentials are independent for each capability. The [database guide](DATABASE.md) describes application persistence and CRUD; the [README](../README.md) covers installation.

## Common contracts and resource ownership

[Provider contracts](../app/providers/contracts/__init__.py) expose:

- `EmbeddingProvider.embed(texts) -> list[list[float]]`, `model_name`, and `close()`.
- `InferenceProvider.chat(messages, max_tokens=512, temperature=None) -> ChatResult` and `close()`.
- `ModelDiscovery.list_models() -> list[ModelInfo]` and `close()`.

Chat messages contain a text `content` and a `system`, `user`, or `assistant` role. `ChatResult` contains `text`, the reported `model`, `finish_reason`, and optional token `usage`. The provider interface is non-streaming plain inference: it does not retrieve documents, store chat history, call tools, or accept images/audio. The application-level chat operation adds document retrieval and grounding before calling this interface.

Operations borrow providers and never close them. Configured entrypoints own and close their clients/models, including failure paths, unless a caller supplies a local model session. Only the requested capability is constructed: ingestion/search does not load the inference model. Local inference uses a separate `llama_cpp.Llama` instance from embeddings.

Remote adapters use the OpenAI Python SDK with automatic retries disabled and a default 60-second request timeout. Their exceptions omit raw server diagnostics, credentials, and prompts. The llama-server adapter uses the server's OpenAI-compatible endpoints; it neither launches a server nor uses a local model instance for HTTP calls. See the [llama-cpp-python HTTP client example](https://llama-cpp-python.readthedocs.io/en/latest/server/).

## Code organization and imports

Model infrastructure lives under `app.providers`. Capability contracts describe what
operations can call; concrete adapters handle local runtimes or remote protocols.

| Namespace | Responsibility |
| --- | --- |
| `contracts.embedding`, `contracts.inference`, `contracts.discovery` | `EmbeddingProvider`, `InferenceProvider`, and `ModelDiscovery` interfaces |
| `types.chat`, `types.model` | Frozen `ChatMessage`, `ChatResult`, and `ModelInfo` values |
| `local.embedding`, `local.inference` | Separate local embedding and inference adapters |
| `openai`, `llama_server` | Backend-specific embedding, inference, and discovery classes |
| `openai_compatible` | Shared HTTP request/response implementations |
| `config` | `ProviderSettings`, endpoint-only `DiscoverySettings`, and pure validation |
| `resolution` | Application settings and catalog-default resolution |
| `factory`, `lifecycle`, `session` | Construction, owned cleanup scopes, and reusable local models |
| `catalog`, `discovery`, `validation`, `normalization` | Catalog I/O, list merging, request validation, and response decoding |

Import interfaces and values from their owning modules:

```python
from app.providers.contracts.embedding import EmbeddingProvider
from app.providers.contracts.inference import InferenceProvider
from app.providers.contracts.discovery import ModelDiscovery
from app.providers.types.chat import ChatMessage, ChatResult
from app.providers.types.model import ModelInfo
```

Contracts do not import SDKs, configuration, application models, or concrete adapters.
The embedding interface remains an ABC; inference and discovery remain protocols.
There is no universal provider base class. Database and storage contracts remain in
their own subsystems.

New composition code uses `create_embedding_provider()`, `create_inference_provider()`,
or `create_discovery_provider()` from `app.providers.factory`. These return only the
requested capability. Use `open_capability_provider(settings)` for an owned embedding
or inference scope and `open_discovery_provider(settings)` for an owned discovery
scope. Operations borrow providers and do not close them.

Discovery settings contain only provider, base URL, credential environment-variable
name, and timeout. `discovery_settings_for(application, capability)` resolves these
without selecting or inventing an inference model. Application listing still queries
each endpoint/credential pair once and keeps catalog-only listing offline.

Compatibility imports remain supported: `app.providers.contracts` exports the same
interfaces and values, `app.embeddings.base` and `app.embeddings.local` alias the
canonical classes, and `app.providers.local` exports `LocalInferenceProvider`.
The old `config.settings_for()` and `catalog.resolve_application()` functions forward
to `resolution`. The original `remote.OpenAIProvider` and `remote.LlamaServerProvider`
classes remain combined delegating facades. `create_provider()` and `open_provider()`
retain those combined remote adapters, including `list_models()`, for existing callers.
Their factory dependency hooks and ownership behavior are unchanged.

## Reusing local models

Use `LocalModelSession` to share loaded models across Python operations. Loading is lazy: creating a session loads nothing, and embedding alone does not load inference. Optional `warmup()` accepts resolved `ProviderSettings` for whichever capabilities you want ready before requests.

```python
from uuid import UUID

from app.application import show_application, ingest_file, search_documents, chat
from app.providers.resolution import settings_for
from app.providers.types.chat import ChatMessage
from app.providers.session import LocalModelSession

application_id = UUID('00000000-0000-0000-0000-000000000001')  # Replace with your local application ID.
application = show_application(application_id=application_id)

with LocalModelSession() as models:
    # Optional; omit inference warmup if the application has no inference model.
    models.warmup(settings_for(application, 'embedding'), settings_for(application, 'inference'))
    documents = ingest_file(path='samples/sample.txt', chunk_size=500, stride=400,
                            application_id=application_id, model_session=models, name='Example document')
    matches = search_documents(query='What is this about?', application_id=application_id,
                               model_session=models)
    answer = chat(messages=[ChatMessage('user', 'Say hello.')], application_id=application_id,
                  model_session=models)
```

`embed_file`, `embed_text_file`, and the legacy `ingest_document` helper also accept `model_session`. Without one, each call continues to load and close its own provider. Remote providers ignore the supplied local session and retain their existing lifecycle. No database or storage resources move into the session.

A session retains one successful load per capability, canonical absolute model path (resolving symlinks), context size, GPU layer count, and inference chat format. Matching settings across applications reuse the same instance. Embedding and inference remain separate instances, even for the same file. Prefixes, embedding-size validation, temperature, and token limits remain operation-specific. Settings for `warmup()` must already have aliases resolved; configured entrypoints resolve aliases before borrowing models.

Concurrent first use constructs a matching provider once; each provider serializes its own model execution. Failed loads can be retried. Successfully loaded providers remain cached after an operation or later warmup fails. Session shutdown rejects new work, waits for active leases, and attempts every model's cleanup once. Repeated `close()` calls are safe, and closed sessions cannot be reopened. Cleanup errors preserve a primary operation exception, or raise `ProviderCleanupError` (a `ProviderError` subclass) if no other exception is active. See [exception contracts](EXCEPTIONS.md) for initialization, request, response, configuration, and session-state errors.

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

Configuration precedence is: explicit allowed CLI override, saved application options, catalog defaults copied when saving, then provider defaults. The standalone Python `embed_file()` API resolves catalog aliases on each call. Chat uses the saved application configuration. Ad-hoc paths remain supported. New applications created through the CLI or configured Python entrypoint store direct local model paths as absolute paths resolved from the creation directory (including `~` and symlinks). Unchanged application paths and direct CRUD service inputs retain their stored values. Configured updates normalize changed local model paths; remote model IDs are never converted into filesystem paths. Local model identity uses the resolved filename, including for symlink overrides.

## Application provider configuration

For a complete local setup including PostgreSQL, object storage, and a model catalog, see the [configuration examples](../README.md#configuration-examples).

Store options under `Application.data.provider_config`, with `embedding` and `inference` sections. `applications:create --data` and `applications:update --data` accept the entire JSON data object; updates replace that object rather than patching nested keys. Keep unrelated metadata when replacing it.

Application chunking is stored separately under `data.chunking` (`chunk_size` and `stride`), defaulting to 500 and 400. Use `applications:create` or `applications:update` with `--chunk-size` and `--stride` to set these values; explicit flags override corresponding values in `--data`. Chunking changes are allowed while documents exist and only affect future uploads.

Common options are `base_url`, `api_key_env`, and `timeout`. Local runtime options are `context_size` (default 2048) and `gpu_layers` (default 0; -1 is accepted for all layers). Local inference also accepts `chat_format`; omit it to use llama-cpp-python's model template selection. Local runtime options have no effect on remote server execution; configure those servers separately.

Embedding options are `document_prefix`, `query_prefix`, and `dimensions`. The dimensions request is omitted unless configured; if supplied it must equal `Application.embedding_size`. The selected upstream model must support it. Regardless of requested dimensions, every returned vector is validated against the application's size. See [OpenAI embedding parameters](https://developers.openai.com/api/reference/resources/embeddings/methods/create).

Local applications without explicit prefixes retain the existing EmbeddingGemma document/query prefixes. Remote applications default to empty prefixes. Configure the appropriate prefixes for other local models, including empty strings where none are required. These formatting choices are model-specific, not transport-specific. Existing documents are never silently re-embedded.

Application updates can change embedding prefixes, requested dimensions, model, embedding provision type, or embedding size while documents exist. Run `documents:process --document-id UUID` for each affected document. Search excludes incompatible stored embeddings until successful reprocessing. Inference settings, endpoint locations, and credential references remain editable; an endpoint change must still serve the configured model.

### Local

```sh
rypiq applications:create --name 'Local example' \
  --embedding-model @embeddinggemma --embedding-size 768 \
  --inference-model @local-chat
```

Local models must be existing GGUF files. `--model` on application-scoped search overrides the path only when its filename matches the selected local model; it cannot switch the application's embedding model. `--context_size` and `--gpu_layers` override local search runtime settings only when explicitly supplied. `documents:create` uses saved application settings and accepts none of these overrides.

### Existing llama.cpp server

```sh
rypiq applications:create --name 'Server example' --embedding-provision-type llama-server --inference-provision-type llama-server \
  --embedding-model embedding-alias --embedding-size 768 --inference-model chat-alias \
  --data '{"provider_config":{"embedding":{"base_url":"http://localhost:8081/v1"},"inference":{"base_url":"http://localhost:8082/v1"}}}'
```

Use the model IDs/aliases served by your endpoints, including `/v1` in each base URL. Embedding and chat may use the same URL or separate servers. RyPiq calls `/embeddings`, `/chat/completions`, and `/models` relative to these base URLs. Servers must support the requested endpoints and embedding pooling/output shape.

Authentication is optional. If required, set `api_key_env` in the appropriate section to an environment-variable name such as `LLAMA_SERVER_API_KEY`, then export that variable. No actual credential belongs in application JSON or the catalog. URLs cannot contain embedded credentials, query parameters, or fragments.

### OpenAI

```sh
export OPENAI_API_KEY='your-provider-key'
rypiq applications:create --name 'OpenAI example' --embedding-provision-type openai --inference-provision-type openai \
  --embedding-model @openai-embedding --embedding-size 768 \
  --inference-model YOUR_CHAT_COMPLETIONS_MODEL_ID
```

The default base URL is `https://api.openai.com/v1`; the default credential reference is `OPENAI_API_KEY`. Override them separately per capability if needed. `Application.api_key` is a RyPiq application credential and is never sent to upstream providers.

Model IDs are not hard-coded to a whitelist. Inference requires a model supporting the Chat Completions endpoint; models exclusive to other endpoints are outside this contract. `max_tokens` maps to OpenAI's `max_completion_tokens` and llama.cpp's `max_tokens`. Temperature is omitted unless supplied, because model support varies. See the [Chat Completions reference](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

## Updating provider configuration

The application provider fields and flags are independent. Omitted update fields stay unchanged, including the other capability's model and settings. Updating a provider does not automatically replace its model or clear old options. Supply compatible values together. Invalid updates fail without saving the changed application. `--clear-inference-model` removes the model but retains `inference_provision_type`.

For an existing database, stop callers and run `rypiq db:migrate` after deploying this version. Existing providers are copied into both fields. Old application code using `provision_type` is incompatible with the new schema; see [database upgrade details](DATABASE.md#independent-provider-upgrade).

```sh
rypiq applications:show APPLICATION_UUID
# Change inference to a local model; embedding settings are retained.
rypiq applications:update APPLICATION_UUID \
  --inference-provision-type local \
  --inference-model /absolute/path/to/chat-model.gguf
# Or use OpenAI; provide OPENAI_API_KEY in the process environment.
rypiq applications:update APPLICATION_UUID \
  --inference-provision-type openai --inference-model YOUR_OPENAI_MODEL_ID
```

OpenAI defaults to its public endpoint and `OPENAI_API_KEY` only if explicit inference options do not override them. When switching from another provider, inspect `data.provider_config.inference` and remove or replace stale endpoint/credential options. Local sessions cache only local capabilities; remote capabilities manage their own clients.

For llama-server, set an inference base URL and the model ID served there. This complete `--data` example assumes the existing embedding settings, chunking, and metadata are exactly those shown. **Adapt it to retain all of your application's existing data**, obtained with `applications:show`; `--data` replaces the entire object.

```sh
rypiq applications:update APPLICATION_UUID \
  --inference-provision-type llama-server --inference-model served-chat-model \
  --data '{"purpose":"knowledge base","chunking":{"chunk_size":500,"stride":400},"provider_config":{"embedding":{"context_size":2048,"gpu_layers":0},"inference":{"base_url":"http://localhost:8080/v1","api_key_env":"LLAMA_SERVER_API_KEY"}}}'
```

Set `LLAMA_SERVER_API_KEY` in the environment if authentication is required; otherwise omit `api_key_env`. Never put credential values in application JSON.

Python callers can preserve existing data while replacing just the inference section explicitly:

```python
from copy import deepcopy
from dataclasses import replace
from uuid import UUID
from app.application import show_application, update_application

value = show_application(application_id=UUID('00000000-0000-0000-0000-000000000001'))
data = deepcopy(value.data)
data.setdefault('provider_config', {})['inference'] = {
    'base_url': 'http://localhost:8080/v1',
    'api_key_env': 'LLAMA_SERVER_API_KEY',
}
updated = update_application(application=replace(
    value,
    inference_provision_type='llama-server',
    inference_model='served-chat-model',
    data=data,
))
```

To switch that example to OpenAI defaults, set `inference_provision_type='openai'`, select its model ID, and replace the inference section with `{}`. The API receives a complete application object; the explicit copy/edit above preserves unrelated data. Application updates retain API keys and creation timestamps and advance the update timestamp.

Verify the result and exercise RAG:

```sh
rypiq applications:show APPLICATION_UUID
rypiq chat 'What does the guide say about setup?' --application-id APPLICATION_UUID
```

Inference-only updates preserve indexed documents. Changes to the embedding provider, model, dimensions, or preprocessing require reprocessing; incompatible embeddings are excluded from search until replacement succeeds. For example, with compatible embedding options already saved:

```sh
rypiq applications:update APPLICATION_UUID \
  --embedding-provision-type local \
  --embedding-model /absolute/path/to/embed.gguf --embedding-size 768
rypiq documents:list --application-id APPLICATION_UUID
# Repeat for each document returned by the listing:
rypiq documents:process --document-id DOCUMENT_UUID
```

When changing dimensions, also update or remove any explicit `data.provider_config.embedding.dimensions` so it matches. Review old embedding endpoint/preprocessing options before switching providers. Inference configuration stays unchanged.

## Chat and discovery commands

```sh
rypiq chat 'What does the guide say about cosine similarity?' --application-id APPLICATION_UUID
rypiq models:list
rypiq models:list --application-id APPLICATION_UUID --discover
rypiq applications:update APPLICATION_UUID --inference-model @local-chat
rypiq applications:update APPLICATION_UUID --clear-inference-model
```

Chat requires an application ID and accepts no model or generation flags. It retrieves the top five application-scoped chunks before inference, with a default maximum of 512 output tokens and provider-default temperature. A missing application inference model fails before constructing a provider. Chat, model discovery, and application show/update return JSON. `applications:list` defaults to text; use `--output json` for JSON. Application list/show/update output omits the application's API key.

Without `--discover`, model listing is catalog-only and performs no remote calls. An application ID filters the catalog to either selected provider. Discovery requires an application ID and queries each distinct remote provider/endpoint/credential combination once; a separate inference endpoint can be listed before choosing its inference model. Local discovery only checks catalog file availability and does not load weights or scan directories.

Results are deterministic and deduplicated by provider and resolved model. They include alias/ID, resolved model, capabilities, source (`catalog`, `discovered`, or `catalog+discovered`), and availability (`available`, `missing`, or `unverified`). Remote discovered capabilities remain empty unless declared in the catalog; names are not used to guess capabilities. A catalog declaration is not proof that a remote model is currently accessible. Discovery never changes the catalog. See [OpenAI model listing](https://developers.openai.com/api/reference/resources/models/methods/list).

## Python usage

```python
from uuid import UUID
from app.application import chat, list_models
from app.providers.types.chat import ChatMessage

result = chat(
    application_id=UUID('00000000-0000-0000-0000-000000000001'),
    messages=[ChatMessage('user', 'What does the guide say about setup?')],
    max_tokens=512,  # Python generation options remain available.
)
for source in result.sources:
    print(source.indicator, source.document_id, source.name, source.filename, source.source)
models = list_models()  # Catalog only.
```

For caller-owned resources, inject the application, embedding repository, embedding provider, and inference provider into `app.operations.inference.chat.Chat` and call `execute()`. The operation borrows these resources and never closes them. Ingestion and search retain their existing injected operations and configured entrypoints. Application chat returns `RagChatResult` from `app.models.chat`, extending `ChatResult` with a list of `ChatSource` values. Each has `document_id` (a Python UUID), `name`, `filename`, `source`, and `indicator` (such as `[1]`). Sources identify all retrieved documents, deduplicated by ID in first-match order, even if the answer reports insufficient evidence. No matches yields an empty list. Provider-level chat results and document embedding snapshots remain unchanged. `app.application.ingest_document()` remains the legacy local composition helper; use `IngestDocument` or configured `ingest_file()` for remote applications.

## RAG prompt customization

The default system prompt is `RAG_SYSTEM_PROMPT` in [app/operations/inference/prompts.py](../app/operations/inference/prompts.py). The pure `build_rag_messages(messages, results)` function adds this prompt and attaches ranked reference passages to the final user message. Passages include document names and IDs, citation indicators, chunk IDs, chunk indexes, and text. Indicators are assigned per unique document in first-match order, starting at `[1]` for each response, and match `result.sources`. The default prompt requests Markdown answers and inline indicators immediately after quotations and sourced claims. It distinguishes verbatim quotes from paraphrases and prohibits invented citations, including citations when retrieval is empty. These instructions are not enforced by output validation. Earlier messages are preserved without modifying the caller's list; the conversation must end with a nonempty user message. Retrieval uses that final message, without query rewriting.

The prompt instructs the model to use only supplied evidence and acknowledge insufficient context. Retrieved text is treated as reference data, not instructions. Empty retrieval explicitly supplies no passages and still invokes inference. Grounding is model-directed, not a correctness guarantee. Provider errors propagate, including context-window overflow; this workflow does not truncate or budget tokens automatically.

A web service or other Python caller can reuse the operation with already configured resources:

```python
from app.operations.inference.chat import Chat
from app.operations.inference.prompts import build_rag_messages
from app.providers.types.chat import ChatMessage

# application_record, embedding_repository, and both providers are caller-owned.
result = Chat(
    application=application_record,
    repository=embedding_repository,
    embedding_provider=embedding_provider,
    provider=inference_provider,
    messages=[ChatMessage('user', 'What does the guide say about setup?')],
    prompt_builder=build_rag_messages,
).execute()
```

To customize formatting or policy, inject a callable with signature
`(messages: list[ChatMessage], results: list[SearchResult]) -> list[ChatMessage]`.
`SearchResult` is defined in `app.models.search`; results retain retrieval rank.
The builder receives a fresh message list and must return a valid chat request. Custom builders that include citations should use `build_chat_sources(results)` from `app.operations.inference.sources` to match the structured response indicators.
Keep resource construction, configuration, and response formatting at the calling boundary.
For plain inference without RAG, call `InferenceProvider.chat()` directly.

## Validation

Provider unit tests cover configuration, catalog resolution, local runtime interaction, normalization, and cleanup. Loopback HTTP tests exercise the actual OpenAI SDK without API credits:

```sh
env/bin/python -m pytest tests/unit/embeddings tests/unit/operations -q
env/bin/python -m pytest tests/integration/embeddings/test_http_providers.py -q
```

Database and existing local embedding integrations use the prerequisites described in [Database](DATABASE.md#database-integration-tests). Live remote services are not contacted by ordinary tests. Local inference validation needs an existing chat-capable model; set `INFERENCE_MODEL_PATH` for the opt-in test. No model is downloaded to run tests.

To explicitly run live remote checks, set `RYPIQ_LIVE_OPENAI_CONFIG` or `RYPIQ_LIVE_LLAMA_SERVER_CONFIG` to a JSON file containing `embedding_model`, `embedding_size`, `inference_model`, and `data.provider_config` as described above. Export the referenced credential variables separately. Then run `tests/integration/embeddings/test_live_remote_providers.py`. Setting these paths opts into outbound provider requests; OpenAI requests may incur charges. Without them, the live checks are skipped and only loopback HTTP compatibility is verified.
