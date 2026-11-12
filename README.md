# `RyPiq`

A Python based RAG service engine for running embedding models.

## Tech Stack

* Python 3.x
* PostgreSQL with pgvector

## Setup

## TODO

* Certain local embedding models have recommended text format. Allow adapting to different formats depending on local embedding model provider.

## CLI Demo Flow

Run these commands from the project root with `rypiq` installed in your active
Python environment. Have both local GGUF files available: an embedding model
and a generation/chat model. The examples use the models in `models/`.

### 1. Initialize the database

```bash
rypiq db:create
rypiq db:migrate
```

### 2. Save a file and create its embeddings

Use `ingest` to save a UTF-8 text file to the configured object storage, split
its content into chunks, generate embeddings, and index the resulting documents
in the database:

```bash
rypiq ingest samples/sample.txt \
  --chunk_size 500 \
  --stride 400 \
  --model models/embeddinggemma-300m-Q4_0.gguf
```

This creates chunks of up to 500 characters, advancing 400 characters between
chunks. The command displays `indexed_chunks` and the saved file's `storage_key`.
Replace `samples/sample.txt` with your own file and repeat for additional files.
`storage:save` alone only stores a file; use `ingest` to also create searchable
embeddings.

You can inspect saved files and check retrieval before generating an answer:

```bash
rypiq storage:list --prefix documents
rypiq search "What is this document about?" \
  --limit 3 \
  --model models/embeddinggemma-300m-Q4_0.gguf
```

### 3. Generate an answer from the indexed documents

```bash
rypiq infer \
  --model models/Qwen3.5-0.8B-UD-Q6_K_XL.gguf \
  --embedding-model models/embeddinggemma-300m-Q4_0.gguf \
  --top-k 3 \
  --query "What is this document about?"
```

`--model` is a local **generation/chat GGUF**, separate from the embedding model.
Retrieval defaults to `models/embeddinggemma-300m-Q4_0.gguf`. If ingestion used
another embedding model, pass that same file with `--embedding-model`.
The generation model's embedded chat template is used by `llama-cpp-python`.

The command prints JSON with `response` and `documents`. Documents are the top-k
indexed chunks selected by semantic search, in relevance order, with citation
numbers, IDs, distances, full content, storage keys, filenames, and chunk indices.
These existing documents become the system prompt's reference material; inference
does not create or persist new documents. The prompt requests grounded answers
with citations such as `[1]`, and an explicit admission when the answer is absent.
With an empty index, the command returns a no-documents message without generating.

Optional controls are `--context-size` (default 4096), `--max-tokens` (default 512),
and `--gpu-layers` (default 0, CPU). `--top-k` controls retrieval, not token sampling.
If the retrieved content exceeds the model's context window, the command reports
an error; reduce the chunk size or top-k, or increase the context size supported
by your model. Full source content is never silently truncated.

### Tests with local models

```bash
env/bin/python -m pytest tests/unit/test_inference.py tests/integration/test_inference_cli.py -q
```

The real-model test uses local
`models/Qwen3.5-0.8B-UD-Q6_K_XL.gguf` and
`models/embeddinggemma-300m-Q4_0.gguf` by default. Override these with
`INFERENCE_MODEL_PATH` and `EMBEDDING_MODEL_PATH`. It creates an isolated Chroma
index, embeds four documents, retrieves three, and verifies that the generated
answer contains a code present only in the source material. It skips when either
model file is unavailable and never downloads models. Other inference tests use
controlled model responses to cover prompt construction, CLI output, validation,
empty retrieval, and generation failures.

## HTTP API

Configure `INFERENCE_MODEL` in `.env`, then start the FastAPI development server
on port 3000:

```bash
rypiq server --port 3000 --reload
```

The server is available at `http://127.0.0.1:3000`, with interactive API
documentation at `http://127.0.0.1:3000/docs`.
Use `POST /query?top_k=3` with a JSON body such as
`{"query": "What is this document about?"}`. No authentication is required.

With the server running, test the endpoint from another terminal:

```bash
curl -X POST 'http://127.0.0.1:3000/query?top_k=3' \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is this document about?"}'
```

See [docs/API.md](docs/API.md) for setup, ingestion, requests, responses, and tests.

### Test with different documents and top-k values

The five text files in [samples/query-demo](samples/query-demo) describe a
fictional Harbor Learning Center: a library, community garden, robotics
workshop, cooking class, and astronomy club. Each has distinct schedules,
costs, contacts, and booking codes so you can check answers against the sources.

After initializing the database as shown above, ingest these files from the
project root before starting the server:

```bash
for file in samples/query-demo/*.txt; do
  rypiq ingest "$file" \
    --chunk_size 500 \
    --stride 500 \
    --model models/embeddinggemma-300m-Q4_0.gguf
done
```

Each file is under 500 characters, so this creates one chunk per file. Configure
the server to use the same embedding model as ingestion, and start it with
`rypiq server --port 3000 --reload`. For the clearest comparison, use an index
containing only these fixtures; other indexed documents can affect retrieval.

Try a focused lookup with `top_k=1`:

```bash
curl -X POST 'http://127.0.0.1:3000/query?top_k=1' \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is the Harbor Learning Center robotics workshop booking code and price?"}'
```

Expected facts: `BOT-731` and 350 credits per person, including materials.

Try a comparison with `top_k=3`:

```bash
curl -X POST 'http://127.0.0.1:3000/query?top_k=3' \
  -H 'Content-Type: application/json' \
  -d '{"query": "When do the Harbor Learning Center garden, robotics workshop, and cooking class meet, and what does each cost?"}'
```

Expected facts, if all three relevant chunks are retrieved: garden on Saturdays,
7 AM to 11 AM, free; robotics on Wednesdays, 4 PM to 6 PM, 350 credits; cooking
on Fridays, 5 PM to 7 PM, 200 credits.

Run the same broad query with `top_k=1`, `3`, and `5` to compare coverage:

```bash
for top_k in 1 3 5; do
  printf '\n--- top_k=%s ---\n' "$top_k"
  curl -X POST "http://127.0.0.1:3000/query?top_k=$top_k" \
    -H 'Content-Type: application/json' \
    -d '{"query": "List every Harbor Learning Center activity with its contact and booking code."}'
  printf '\n'
done
```

With only these five chunks indexed, `top_k=5` should supply all of these facts:

| Activity | Contact | Booking code |
| --- | --- | --- |
| Library | Mira Santos | BOOK-214 |
| Community garden | Leo Cruz | GROW-582 |
| Robotics workshop | Ana Reyes | BOT-731 |
| Cooking class | Nico Lim | COOK-469 |
| Astronomy club | Sara Tan | STAR-926 |

Inspect both `response` and `documents` in each JSON result. `top_k` controls
the maximum number of retrieved chunks, not the number of answers or distinct
files. Smaller values provide less source material; semantic ranking does not
guarantee that a comparison retrieves every activity named in the question.
Check the returned filenames, content, and citations against the expected facts.

Finally, test a question whose answer is absent from all five files:

```bash
curl -X POST 'http://127.0.0.1:3000/query?top_k=5' \
  -H 'Content-Type: application/json' \
  -d '{"query": "What is the Wi-Fi password at Harbor Learning Center?"}'
```

The answer should acknowledge that the documents do not provide the password.
