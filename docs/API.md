# Rypiq API

The FastAPI application exposes `POST /query` without authentication. It uses
an app factory, route registration, and a controller delegating to the existing
`infer` operation, following the structure of
[default_api_fast](https://github.com/cloudband-solutions/default_api_fast).

## Start the development server

Run from the project root:

```bash
source env/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Edit `.env` to point to local GGUF files:

```dotenv
INFERENCE_MODEL=models/Qwen3.5-0.8B-UD-Q6_K_XL.gguf
EMBEDDING_MODEL=models/embeddinggemma-300m-Q4_0.gguf
```

`INFERENCE_MODEL` is required and selects the generation/chat model for `/query`.
`EMBEDDING_MODEL` defaults to `models/embeddinggemma-300m-Q4_0.gguf` and must match
the model used to index your documents. Relative paths resolve from the working
directory. The server reads `.env` from that directory; exported environment
variables take precedence. Restart after changing `.env`.

Initialize the database and ingest a UTF-8 text file to save it, create embeddings,
and index its chunks:

```bash
rypiq db:create
rypiq db:migrate
rypiq ingest samples/sample.txt \
  --chunk_size 500 --stride 400 \
  --model models/embeddinggemma-300m-Q4_0.gguf
rypiq server --reload
```

If you use another embedding model, pass its path to `ingest --model` and set
`EMBEDDING_MODEL` to the same path. The server uses the existing database and
storage YAML configuration; it does not automatically migrate or ingest files.

The server listens on `http://127.0.0.1:8000`. Set `--host` and `--port` as needed:

```bash
rypiq server --host 127.0.0.1 --port 8100 --reload
```

Equivalent commands, including when the console script is not installed:

```bash
python -m app.cli server --reload
uvicorn app:create_app --factory --reload
```

Startup validates both model paths, opens the configured document collection,
and loads the embedding model. Each query loads the generation model through
`infer` and releases it afterward. Inference requests are serialized within each
server process to avoid simultaneous local generation model allocations. The
embedding model is released on shutdown. Generation defaults remain those of
`infer`: CPU, a 4096-token context window, and up to 512 output tokens.

Interactive documentation: `/docs`. OpenAPI schema: `/openapi.json`.

## POST /query

Send the question in a JSON body. Pass `top_k` in the URL query string.
No authorization header is required.

```bash
curl -X POST 'http://127.0.0.1:8000/query?top_k=3' \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is this document about?"}'
```

| Input | Location | Requirement |
| --- | --- | --- |
| `query` | JSON body | Required, nonblank string; surrounding whitespace is removed |
| `top_k` | URL query parameter | Integer greater than zero; defaults to 3 |

The body accepts only `query`; model paths are configured on the server.
`top_k` is passed directly to `infer` as the number of chunks to retrieve, not as
a token-sampling parameter. Fewer chunks can be returned when the index contains
fewer documents.

### Successful response — 200

Example response for an index containing a workshop document:

```json
{
  "response": "The Rypiq workshop access code is ORCHID-742 [1].",
  "documents": [
    {
      "citation": 1,
      "id": "59f329a5-b72a-4f69-b599-36b5e3b47c82",
      "distance": 0.15843522548675537,
      "content": "The Rypiq workshop access code is ORCHID-742.",
      "storage_key": "documents/workshop.txt",
      "filename": "workshop.txt",
      "chunk_index": 0
    }
  ]
}
```

`documents` contains ranked indexed chunks used in the system prompt, with full
content and source metadata. Citation numbers correspond to the prompt's source
numbers. The prompt asks the model to answer from these documents, cite its
sources, and say when evidence is insufficient. Inference does not create new
database records.

An empty index returns 200 without running generation:

```json
{
  "response": "No indexed documents matched the query.",
  "documents": []
}
```

### Errors

| Status | Meaning |
| --- | --- |
| 422 | Invalid or missing JSON body/query, or invalid `top_k` |
| 400 | Inference cannot process the input, such as context-window overflow; try a smaller `top_k` or shorter question |
| 503 | Local inference fails with a model, filesystem, or runtime error |

Errors use FastAPI's `detail` field. Runtime details are logged on the server.
Missing configuration or model files and unavailable database collections fail
startup. A corrupt or unsupported generation GGUF fails when a query loads it.

## Tests

```bash
env/bin/python -m pytest tests/integration/test_query_api.py -q
```

Tests cover `.env` loading and environment precedence, request validation,
`top_k` forwarding, response documents, errors, cleanup, and server CLI options.
A real local-model test uses an isolated Chroma index and verifies a grounded
answer through HTTP. Set `INFERENCE_MODEL_PATH` and `EMBEDDING_MODEL_PATH` to
override its model paths; it skips if either file is unavailable.
