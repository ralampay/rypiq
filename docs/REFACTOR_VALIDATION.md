# CLI/backend refactor validation

Final status: **SUCCESS** after four TDD iterations.

## Delivered execution paths

- CLI: parser → explicit handlers → configured application/database entrypoints → shared operations → repositories/providers.
- Configured Python: `app.application.embed_file`, `ingest_file`, `search_documents`, and `execute_database_command(str)` return Python values and manage their own resources.
- Injected Python: `EmbedTextFile`, `IngestDocument`, and `SearchDocuments` expose `execute()` and borrow caller-owned dependencies. Existing function entrypoints remain supported.
- Search no longer constructs object storage. Model cleanup and ingestion compensation preserve primary failures. Standalone repository callers have an explicit cleanup path.
- Local object publication is atomic and exclusive; failure/concurrency coverage verifies the final key never exposes partially copied data or overwrites a competing writer.

See [execution paths](EXECUTION_PATHS.md) for examples, extension points, and ownership rules.

## Iteration evidence

| Iteration | Initial failure evidence | Resolution and validation |
| --- | --- | --- |
| 1: command and CLI boundaries | Six new tests failed: no injected command classes and no explicit-argv CLI entrypoint. | Extracted parser/handlers; added embedding, ingestion and search commands plus compatibility wrappers. Focused suite: 35 passed. Full suite: 139 passed, but the coverage gate correctly failed at 89.68%. |
| 2: resource ownership | Three failing tests showed search required storage configuration and provider cleanup could mask the primary error or omit partial-completion context. | Added database-only composition and safe provider cleanup. Corrected an intermediate argument-wiring error caught by the tests. Operations and CLI failure tests: 27 passed. |
| 3: meaningful failure coverage | An injected final-copy failure left a partial stored object. The new publication-boundary test also failed because exclusive atomic publication was absent. | Replaced final copying with same-directory staging and exclusive hard-link publication; tested visibility, competing writers, cleanup failures, retry, and missing paths. Added configuration and operation failure tests. Full suite: 186 passed, 96.46% coverage. |
| 4: Python database execution and final review | Two tests failed because a namespace-free configured database entrypoint was missing. | Added `execute_database_command(str)`, retained legacy adapters, verified validation before construction, and updated execution documentation. Final full suite: **189 passed**, zero failures or skips. |

The temporary final-copy reproducer was removed after recording its failure because the refactor eliminated that copy operation. Permanent tests exercise the new staging/publication boundaries and externally observable guarantees.

## Acceptance results

| Acceptance criterion | Result | Evidence |
| --- | --- | --- |
| Clear CLI path with parsing/presentation separate from business logic | PASS | Explicit-argv tests, handler tests, and fresh-process installed-CLI integration tests. |
| Direct Python operations without argparse, stdout, or process exits | PASS | Injected-command tests, configured-facade tests, string-based database execution, and documented call chains. |
| Preserve factories, repository/provider interfaces, supported CLI contracts and function APIs | PASS | Existing regression suite plus compatibility-adapter tests; actual PostgreSQL/pgvector and local-model pipeline tests. |
| Explicit resource ownership and error handling | PASS | Borrowed-resource assertions; cleanup on setup/execution failure; compensation failures preserve the original exception and attempt remaining cleanup. |
| Meaningful coverage above the 90% gate | PASS | Entire `app` package: 96.48% combined statement/branch coverage; 95.50% branch coverage. No added exclusions or application-file omissions. |
| Regression and review checks | PASS | 189 tests pass; compilation, dependency consistency, whitespace checks, documentation links, and example syntax checks pass. |

## Final coverage

| Metric | Result |
| --- | --- |
| Statements | 879 / 909 = **96.70%** |
| Branches | 191 / 200 = **95.50%** |
| Combined gate metric | **96.48%**; required minimum **90%** |
| CLI entrypoint, parser, handlers | **100%** each |
| Configured application layer | **100%** |
| All operation modules | **100%** each |
| Local storage provider | **100%** |

Coverage includes subprocesses used by installed CLI tests. The initial baseline was 133 passing tests with approximately 82% parent-process coverage; it did not yet include subprocess coverage, so it is not a directly comparable before/after percentage. The final report includes all application files and has zero excluded lines. HTML output is generated locally in `htmlcov/index.html` and is ignored by Git.

Remaining uncovered lines include abstract interface stubs, database-creation/retry error paths, an undefined stored cosine-distance guard, and several invalid provider/text-file input cases. These are disclosed coverage gaps, not claims of exhaustive verification or known failing behavior. No tests were added solely to execute abstract stubs.

## Commands and environment

The project `env` environment was used with an isolated PostgreSQL 17 cluster, installed pgvector extension, and the existing local GGUF model. Tests created uniquely named temporary databases. Application databases were not used. No model downloads were performed. The disposable server was stopped after validation.

`RYPIQ_TEST_ADMIN_URL` pointed to the isolated server on `127.0.0.1:55439`. The final validation command was:

```sh
env/bin/python -m pytest tests --cov=app --cov-branch \
  --cov-config=pyproject.toml --cov-report=term-missing \
  --cov-report=json:/tmp/rypiq-coverage-final.json --cov-report=html -q
```

Result: **189 passed in 20.50 seconds**, with the coverage gate satisfied.

Additional checks:

```sh
env/bin/python -m compileall -q app tests
env/bin/python -m pip check
git diff --check
```

Compilation and whitespace checks succeeded. Dependency validation reported no broken requirements. Local Markdown links and Python documentation examples were checked without executing example side effects. No configured linter/type-checker or benchmark results are claimed.

## Limits

- Existing fixed-schema and PostgreSQL/pgvector constraints remain unchanged; there is no automatic schema alteration or legacy-data import.
- Exact vector search has no approximate index. Ingestion still uses per-record repository transactions and best-effort compensation rather than a distributed transaction.
- Hard-link publication requires filesystem support and does not promise crash/power-loss durability. Cleanup failures after publication leave a committed object and report the partial outcome.
- Coverage is evidence of exercised paths, not proof that every possible behavior is correct. The configured gate uses combined statement/branch coverage; separate branch coverage is reported above.
