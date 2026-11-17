# Fixed-schema refactor validation

Final status: **SUCCESS**. Six TDD iterations completed; maximum allowed: ten.

## Environment and scope

Validation used the project's `env/bin/python` (Python 3.14.3), PostgreSQL 17 on an isolated temporary cluster, the installed pgvector server extension, SQLAlchemy 2.1.8, Psycopg 3.3.9, and pgvector Python 0.4.2. Database tests created unique `rypiq_test_*` databases. No configured application database was changed. The existing local embedding model was available; no model was downloaded.

The agreed fixed-schema refactor superseded the previous migration-engine/storage task sequence. Local-storage atomic-publication changes were not part of this implementation.

## Iterations

| Iteration | Red evidence | Implementation and green evidence |
| --- | --- | --- |
| 1 | 14 failures, 6 passes: dimensions/default validation, renamed model field, adapter restriction, and dispatch differed from the contract. | PostgreSQL configuration validation, `model_name`, and four-command dispatch: 20 passed. |
| 2 | 11 lifecycle failures: the old create operation installed neither pgvector nor tables; migrate/reset/drop behavior was missing. | Explicit schema, transactional initialization, compatibility checks, create/drop/reset: 11 passed on PostgreSQL. The missing-database test also exposed Psycopg connection errors without SQLSTATE; fixed with a sanitized connection diagnostic. |
| 3 | 9 failures, 4 passes: retrieved vectors were arrays, search ranked in Python, and invalid vectors reached the database. | Native SQL cosine search, list conversion, and validation before I/O: 13 passed. |
| 4 | 8 failures: obsolete dependencies/scaffolding remained and CLI command registration differed. | Removed obsolete implementations and workshop, updated dependencies/CLI/config. A full run exposed duplicate test module names after reorganization; pytest importlib mode fixed collection. Full suite: 117 passed. |
| 5 | 3 failures, 3 passes: contexts did not own/close the database, including setup and CLI error paths. | Explicit context ownership and guaranteed CLI cleanup. Focused operations/error suite: 17 passed. |
| 6 | 2 failures, 31 passes: overlong ASCII and Unicode database names were accepted. | Reject names exceeding PostgreSQL's 63-byte identifier limit. Verified initialization rollback, schema mismatch atomicity, ingestion compensation, actual model CLI ingestion/search, and reset preservation. Final full suite: **133 passed, zero failures, zero skips**. |

## Acceptance evidence

| Criterion | Status | Evidence |
| --- | --- | --- |
| Remove migration autogeneration, obsolete backend/commands, workshop, and web scaffolding | PASS | Project-contract tests, removed-command CLI tests, source search, and final diff inspection. |
| PostgreSQL-only environment configuration, credentials, configurable dimensions | PASS | Configuration tests cover defaults, invalid types/ranges, password environment resolution, malformed YAML redaction, protected names, and identifier length; URL test covers `%`, `@`, and `:`. |
| Explicit Document schema with model name and native vectors | PASS | Real extension/table inspection and complete Document round-trip through CRUD; model field tests verify no persistence decorator remains. |
| Repeatable create/migrate; data preservation; incompatible-schema rejection | PASS | Lifecycle and repository tests; mismatch cases cover missing/extra columns, type, nullability, primary key, and vector dimensions. Transaction tests prove no partially created tables/extension remain after failure. |
| Safe target-specific drop/reset and preservation of object storage | PASS | Real lifecycle and end-to-end reset tests; repeat drop, maintenance-database protection, active external connection survival, and partial-reset error reporting. |
| CRUD, pagination, ingestion, filtering, vector validation, and SQL search | PASS | Repository, pipeline, and existing operation tests. SQL capture confirms `<=>`, `ORDER BY`, and `LIMIT` are executed by PostgreSQL. |
| Installed CLI works outside the repository | PASS | Fresh subprocesses use the installed `rypiq` executable, temporary working directories, and configuration overrides for lifecycle and model-backed pipeline tests. |
| Logical test domains and unrelated regressions | PASS | Tests grouped under unit/integration domain directories. All 133 tests collected and passed, including storage, processing, and model tests. |
| Documentation matches implementation | PASS | README/configuration examples reviewed against current CLI help, schema, settings, and executed tests. |
| No known critical issue within the agreed scope | PASS | Focused review corrected resource cleanup and identifier-truncation gaps; full suite, compilation, dependency check, and diff check pass. |

## Commands executed

All pytest commands used the project environment. PostgreSQL runs set `RYPIQ_TEST_ADMIN_URL` to the disposable maintenance database on `127.0.0.1:55439`; the cluster was initialized with UTF-8 and local test authentication.

```sh
env/bin/python -m pytest tests/unit/database -q
env/bin/python -m pytest tests/integration/database/test_lifecycle.py -q --tb=short
env/bin/python -m pytest tests/integration/database/test_repository.py -q --tb=short
env/bin/python -m pytest tests/unit/database/test_project_contract.py tests/integration/database/test_cli.py -q --tb=short
env/bin/python -m pytest tests/unit/operations/test_context.py tests/unit/database/test_errors.py -q --tb=short
env/bin/python -m pytest tests/unit/operations tests/unit/database/test_errors.py tests/integration/operations -q --tb=short
env/bin/python -m pytest tests/unit/database/test_configuration.py tests/integration/database/test_pipeline.py tests/integration/database/test_lifecycle.py -q --tb=short
env/bin/python -m pytest tests -q --tb=short
env/bin/python -m pytest --collect-only -q tests
env/bin/python -m compileall -q app tests
env/bin/python -m pip check
git diff --check
```

Final full run: **133 passed in 17.13 seconds**. Collection: **133 tests**. Compilation and whitespace checks succeeded; dependency check reported no broken requirements. No linter or type-checker configuration existed, and no benchmark claims are made.

The Python pgvector package was installed into `env`, and the editable package metadata was refreshed with `env/bin/python -m pip install -e '.[dev]' --no-deps`. Earlier unrelated packages already present in the shared environment were not uninstalled.

## Deliberate limitations

- No automatic alterations, revision history, rollback command, or import of legacy database contents. Add future tables explicitly; review existing-table changes separately.
- Exact vector search has no approximate index. Use consistent embedding models; a common dimension does not imply model compatibility.
- Drop/reset destroy the selected database's contents. Reset cannot restore data if recreation fails. Object-storage files are preserved.
- Initialization assumes a single schema writer. Integration coverage requires a disposable PostgreSQL server with pgvector; absent prerequisites must be reported as skips/blockers on other machines.
