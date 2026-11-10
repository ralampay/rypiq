# Modular Python application development with RAG

An instructor-ready, one-day workshop for Python developers who know functions, classes, and imports. Participants build **ragworkshop**, an installable package with the **rag-workshop** CLI. The running project ingests TXT, DOCX, and text-based PDF documents, stores originals, embeds chunks through an interface, and retrieves source-bearing results from persistent Chroma.

The required path uses a deterministic **fake embedding provider**, real parsers, local file storage, and real Chroma. It teaches application architecture and retrieval mechanics without claiming the fake demonstrates semantic quality. Real GGUF inference is an explicitly opt-in extension. Generation, OCR, cloud deployment, and a production HTTP API are outside the core day.

## Start here

Learners: begin with [Module 1 discussion](modules/01-introduction/ModuleDiscussion.md), then its activity. Keep one student project throughout the day. Each subsequent module builds on the preceding checkpoint.

Instructors: read the discussion, activity, and solution for each module before class. Use focused TDD exercises rather than asking participants to type every infrastructure file. Provide the CLI/configuration scaffolding, but have learners predict and run the tests before implementing the TODOs. Reveal solutions during review, not at the start of the activity.

| Module | Discussion | Activity | Solution |
|---|---|---|---|
| 1. Introduction and setup | [Read](modules/01-introduction/ModuleDiscussion.md) | [Exercise](modules/01-introduction/Activity.md) | [Review](modules/01-introduction/ActivitySolution.md) |
| 2. Text and embeddings | [Read](modules/02-text-and-embeddings/ModuleDiscussion.md) | [Exercise](modules/02-text-and-embeddings/Activity.md) | [Review](modules/02-text-and-embeddings/ActivitySolution.md) |
| 3. File handling | [Read](modules/03-file-handling/ModuleDiscussion.md) | [Exercise](modules/03-file-handling/Activity.md) | [Review](modules/03-file-handling/ActivitySolution.md) |
| 4. Storage | [Read](modules/04-storage/ModuleDiscussion.md) | [Exercise](modules/04-storage/Activity.md) | [Review](modules/04-storage/ActivitySolution.md) |
| 5. Database and integration | [Read](modules/05-database-and-integration/ModuleDiscussion.md) | [Exercise](modules/05-database-and-integration/Activity.md) | [Review](modules/05-database-and-integration/ActivitySolution.md) |

## Diagram guide

Each module now includes three explained architecture diagrams in plain text, an activity map, and a solution trace. These **25 text diagrams** work in terminal readers and printed handouts alongside the six Mermaid diagrams. Arrow legends distinguish construction, runtime calls, data transformations, and implementation relationships.

| Module | Architectural ideas visualized |
|---|---|
| [1. Introduction](modules/01-introduction/ModuleDiscussion.md#architecture-diagrams-explained) | RAG stages, dependency inversion, package installation and executable dispatch |
| [2. Text and embeddings](modules/02-text-and-embeddings/ModuleDiscussion.md#architecture-diagrams-explained) | Factory/Strategy/Adapter roles, chunk overlap, ordered embedding batches |
| [3. File handling](modules/03-file-handling/ModuleDiscussion.md#architecture-diagrams-explained) | Loader boundaries, page/paragraph provenance, unit versus parser integration tests |
| [4. Storage](modules/04-storage/ModuleDiscussion.md#architecture-diagrams-explained) | Contract injection, logical keys versus physical bundles, atomic publication and cleanup ownership |
| [5. Integration](modules/05-database-and-integration/ModuleDiscussion.md#architecture-diagrams-explained) | Configuration and composition, links between original files and vector records, compensation and retrieval |

Use the discussion diagrams to trace responsibilities before reading code. Ask learners to annotate the activity maps before implementing TODOs; use the solution traces during review. The diagrams introduce no extra implementation requirements and do not replace the test-first walkthroughs.

## Workshop sequence

1. Introduction and project setup.
2. Text handling and embedding abstraction.
3. File handling.
4. Dealing with storage.
5. Database, configuration, and end-to-end integration.

Each module includes teaching, a walkthrough, a core activity, review, a short break, and troubleshooting. Move a learner to the matching checkpoint if setup blocks progress. Stretch work is optional. The discussions are reference readings as well as lesson scripts; instructors should demonstrate the named walkthrough steps rather than narrate every source line.

## Preparation and materials

- Linux, Python 3.12+, venv support, and a terminal/editor. Validation used Python **3.14.3**; other supported interpreter versions were not executed during preparation.
- Install the dev dependencies before class where possible. Package installation needs network access or a prepared wheel cache; **default tests and demos need no model downloads, external services, or credentials**.
- Runtime dependencies: ChromaDB 1.5.9, PyYAML 6.0.3, python-dotenv 1.2.4, python-docx 1.2.0, pypdf 6.12.2. Development: pytest 9.2.0, ReportLab 4.4.10. Optional GGUF: llama-cpp-python 0.3.36.
- All dependencies are declared in the checkpoint `pyproject.toml` from Module 1, avoiding surprise installation work later. Version pins document the tested direct dependencies, not a full transitive dependency lock.
- The 15 module Markdown files are the primary curriculum. `starters/module-NN/` contains only that module's new/replacement files. Apply these overlays in order to the student's completed previous checkpoint.
- `reference/module-NN/` contains a complete cumulative implementation through that module. Later concepts never need to be imported by an earlier checkpoint.
- Generated parser fixtures live in pytest temporary directories. The Module 3 documents also provide exact commands to create manual samples.

**Working directory: the repository root containing `workshop/`.** For an instructor preview, copy a checkpoint to a separate directory rather than changing the reference files:

```bash
export WORKSHOP_MATERIALS="$PWD/workshop"
mkdir -p "$HOME/rag-workshop-preview"
cp -R "$WORKSHOP_MATERIALS/reference/module-05/." "$HOME/rag-workshop-preview/"
cd "$HOME/rag-workshop-preview"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
rag-workshop info
rag-workshop --env test db:create
rag-workshop --env test db:migrate
rag-workshop --env test ingest samples/sample.txt
rag-workshop --env test search python
rag-workshop --env test storage:list
```

Expected: the offline suite passes with one real-model skip; `info` prints `ragworkshop: modular ingestion and retrieval`; ingestion reports one indexed chunk and a generated storage key; search returns the sample with provenance; storage lists the original file. Repeated ingestion creates a separate source by design. Exact paths, IDs, and floating-point distances are not snapshot assertions.

To recover a learner's checkpoint, copy the matching reference into a **new** project directory, activate/install it there, and run its tests. Do not apply the final reference prematurely: that would hide the progressive architecture and erase meaningful Red checkpoints.

## Architecture and contracts

The reference application informed the separation into operations, provider interfaces, factories, object storage, database lifecycle, repositories, migrations, and environment-specific YAML. The workshop deliberately improves several boundaries: a `src` layout and installed-script tests, independent loaders and chunkers, injected embedding instances, typed records, explicit initialization, stable storage metadata, and compensation for partial writes. No existing application is renamed or modified.

- **Domain and operations:** chunk records, extracted segments, stored objects, documents, and results are plain dataclasses. Operations receive dependencies; they do not read environment variables or construct model runtimes.
- **Text preparation:** plain/Markdown/JSON adapters share character chunking. Overlap equals size minus stride. Offsets are code points in the extracted segment; paragraph/page metadata connects them to the source.
- **Embeddings:** a Protocol defines ordered batch output. Validation rejects wrong counts/dimensions, non-finite values, and zero vectors. The GGUF adapter imports its runtime lazily and identifies model bytes plus preprocessing policy.
- **Original files:** ObjectStorage uses logical keys. Local bundles atomically publish bytes plus metadata. Keys and symlinks are checked; collisions never overwrite a stored object. The storage root must be trusted.
- **Vector index:** DatabaseBackend owns lifecycle; DocumentRepository owns record/query mapping. Chroma always receives explicit vectors. One collection uses one embedding identity and dimension. Metadata filters support scalar equality and conjunction.
- **Composition:** the CLI resolves settings and builds a context. Precedence is CLI > environment > dotenv (without overriding environment) > selected YAML section > defaults. YAML-relative paths resolve against the YAML file; explicit relative overrides resolve against the working directory. All application variables start with `RAG_WORKSHOP_`.
- **Failures:** ingestion deletes all allocated chunk IDs and its new stored object on failure. If cleanup fails too, both errors are retained. This is compensation, not a cross-store transaction or crash-recovery system.

Simple factories, registries, Strategy, Adapter, Repository, dependency injection, and a composition root are explained through their actual participants and tradeoffs. The migration callable is a small command record. The workshop avoids abstract framework layers that have no replacement use case.

## Instructor facilitation

| Module | Ask learners | Expected explanation | Recovery focus |
|---|---|---|---|
| 1 | Why test the installed executable outside the source directory? | It catches packaging/import assumptions hidden by the current directory. | Activate the right environment; reinstall after changing entry points. |
| 2 | Why is stride 3 with size 4 an overlap of 1? | Starts advance by 3; the previous final character is reused. | Draw offsets; inspect one vector batch at a time. |
| 3 | Why does the second page's first chunk start at offset 0? | Offsets are segment-relative; page number supplies provenance. | Verify fixture text exists before blaming the chunker. |
| 4 | Why can't callers open a returned payload path? | The contract must work for non-filesystem providers. | Trace logical keys and check collision/cleanup ownership. |
| 5 | Why reject another model with the same dimension? | Equal shape does not imply equal embedding space. | Inspect settings before adapters, then trace IDs through compensation. |

During Red, ask learners to name the intended failure before running pytest. During Green, implement one behavior at a time. During Refactor, ask which tests constrain externally visible behavior. Pair review should explain one tradeoff, not merely identify a named pattern. Keep the final demo deterministic so infrastructure or model variance cannot consume the review time.

## Validation

Validation commands and actual results are recorded below. These are workshop checkpoint tests, not a claim that the host application's existing suite was run.

Validated on Python 3.14.3 with the pinned direct dependencies. Each checkpoint was separately installed editable before its suite ran.

| Checkpoint | Actual Green result | Actual starter Red result |
|---|---|---|
| Module 1 | 3 passed | 2 failed, 1 passed |
| Module 2 | 32 passed, 1 skipped | 9 failed, 23 passed, 1 skipped |
| Module 3 | 40 passed, 1 skipped | 7 failed, 33 passed, 1 skipped |
| Module 4 | 54 passed, 1 skipped | 12 failed, 42 passed, 1 skipped |
| Module 5 | 66 passed, 1 skipped, 1 warning | 11 failed, 55 passed, 1 skipped, 1 warning |

Starter failures are intentional `NotImplementedError` failures, with no test collection errors. Module 1's packaging fields are completed before testing its behavioral Red checkpoint. Green checkpoints have no failures.

Additional completed checks:

- Built `ragworkshop-0.1.0-py3-none-any.whl`, installed it instead of the editable package, and confirmed imports came from the validation environment's `site-packages`.
- Ran the wheel-installed final suite with Python socket/DNS audit events blocked in both pytest and CLI subprocesses: **66 passed, 1 skipped, 1 dependency warning**. The audit guard is not an OS-level sandbox for native libraries.
- Compared all 15 module documents to exercise and solution files: **103 Python code blocks parsed**, with documented exercise/solution files matching their executable counterparts.
- Parsed **all 6 Mermaid diagrams** using Mermaid 11.12.0.
- Checked declared dependencies and console entry points, and confirmed the materials do not use the reference project's name.
- The final suite covers unit behavior, generated DOCX/PDF fixtures, filesystem integration, Chroma persistence/filtering, all three ingestion formats, environment isolation, and installed CLI E2E across separate processes.


**Working directory: the repository root containing `workshop/`.** Use a dedicated validation virtual environment; the scripts reinstall checkpoints sequentially into that environment and leave Module 5 installed. Activate it so subprocess tests find the correct executable.

```bash
python3 -m venv /tmp/ragworkshop-validation
source /tmp/ragworkshop-validation/bin/activate
python -m pip install -e "workshop/reference/module-05[dev]"
python workshop/tools/validate_checkpoints.py
python workshop/tools/validate_starters.py
python workshop/tools/audit_materials.py
python workshop/tools/validate_offline.py
```


For a wheel-install check, still from the repository root with the validation environment active:

```bash
python -m pip wheel --no-deps workshop/reference/module-05 --wheel-dir /tmp/ragworkshop-wheels
python -m pip install --force-reinstall --no-deps /tmp/ragworkshop-wheels/ragworkshop-0.1.0-py3-none-any.whl
python workshop/tools/validate_offline.py
```

The audit compares solution and activity code to the actual checkpoint/overlay files, parses Python examples, checks entry points/dependencies, and extracts Mermaid diagrams. To validate diagram syntax, install tooling outside the repository:

```bash
npm install --prefix /tmp/ragworkshop-mermaid mermaid@11.12.0 jsdom@26.1.0
node workshop/tools/validate_mermaid.mjs /tmp/ragworkshop-mermaid/node_modules
```

Mermaid diagrams were syntax-parsed with Mermaid 11.12.0; visual rendering across every Markdown viewer is not claimed. The code blocks are runnable files or explicitly labeled snippets; JSON output comments describe stable fields where paths/UUIDs vary.

## Limitations and optional work

- The real-GGUF test is provided but was not executed: no compatible model was selected for workshop validation. Its mocked adapter test runs in the default suite. The optional model commands contain an intentional user-supplied model path and dimension.
- The pinned Chroma dependency emits one Python 3.14 deprecation warning concerning `asyncio.iscoroutinefunction`. Tests pass; no private dependency patch is included.
- DOCX support covers body paragraphs, not tables/headers/text boxes. PDF extraction is text-based and does not guarantee complex-layout reading order; OCR and encrypted PDFs are excluded.
- The local adapter and migration/space initialization assume a trusted, single-writer application. Atomic publication does not promise power-loss durability. Failed compensation and process crashes need production reconciliation.
- The fake is lexical and deliberately small; it validates composition, persistence, and retrieval mechanics, not semantic ranking quality. Real-provider document/query formatting and token budgets must follow the chosen model.
- FastAPI authentication and PostgreSQL/pgvector are discussed as later adapters, not executable production integrations. No HTTP service is exposed, and no cloud integration is required.
