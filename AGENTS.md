# Agent instructions

## Purpose and references

RyPiq is a Python CLI and library for processing text, generating embeddings, storing original files, and searching documents with PostgreSQL and pgvector. Keep domain logic independent of entrypoints.

Read [code conventions](docs/CODE_CONVENTIONS.md) before choosing a new abstraction. Use [README.md](README.md) for setup and command contracts and [execution paths](docs/EXECUTION_PATHS.md) for the CLI and direct Python call chains. Inspect nearby code, tests, and any more-specific instructions before editing. Explicit task requirements take precedence over these repository defaults.

## Architecture

- Keep CLI handlers thin: parse arguments, compose dependencies, invoke operations, and format output/errors. Do not put SQL or business workflows in argument handlers.
- Prefer a command object with one public `execute()` method for a new multi-step or stateful business workflow. Use explicit keyword arguments and inject dependencies. Constructors only capture inputs/dependencies; business side effects happen during execution.
- Keep small stateless operations as functions. Existing operation functions remain supported. Do not convert them solely to satisfy a naming convention.
- Commands return typed values and raise exceptions. Preserve existing return types and CLI JSON contracts. Do not introduce a result envelope, universal command base class, or validation framework without a concrete need in the task.
- Factories construct configured implementations; repositories own persistence/SQL; providers implement infrastructure interfaces; domain dataclasses hold data. Choose the pattern by responsibility, not by file naming.
- Compose dependencies at entrypoints or application-context boundaries. Commands borrow injected resources and do not close resources owned by their caller.
- Keep configuration loading and provider selection outside core business logic. Reuse existing factories and interfaces before adding new ones.

## Project constraints

- The only database backend is PostgreSQL with required pgvector support. The explicit schema lives in `app/database/schema.py`; there is no migration-file generation or model discovery.
- Preserve `Document.model_name` and configurable embedding dimensions, defaulting to 768. Keep vector validation and cosine search semantics consistent with the repository.
- Supported database commands are `db:create`, `db:migrate`, `db:drop`, and `db:reset`. Create initializes the database and fixed tables. Migrate creates missing tables and rejects incompatible existing schemas. Drop/reset operate on the selected database and preserve object-storage files.
- Do not introduce web frameworks, additional database backends, migration autogeneration, or speculative abstractions unless the task calls for that architectural change.
- Preserve unrelated functionality and user edits. Do not run destructive database commands against application data as part of development validation.

## Implementation workflow

1. Inspect relevant source, configuration, tests, and existing patterns. Identify the owning layer and observable acceptance criteria.
2. Plan the smallest coherent change. Proceed with routine implementation decisions within the authorized scope; ask only when missing information materially changes behavior or safety.
3. For behavioral changes, follow red–green–refactor: write a behavior test, verify it fails for the intended reason, implement the change, then simplify while tests remain green. Collection/setup errors are not evidence of a valid red test.
4. Test success and meaningful failure paths, including side-effect ordering and resource cleanup. Fix underlying causes; add regression coverage for discovered defects.
5. Run targeted validation first, then relevant regressions. Review the complete diff for incorrect layer boundaries, duplicated logic, accidental changes, leaked credentials, and unhandled failures.
6. Iterate while concrete acceptance gaps remain. Stop when requirements pass or a real blocker prevents further progress; do not churn over cosmetic preferences. Honor task-specific iteration limits.
7. Report what changed, actual validation commands/results, skips, and unresolved issues. Never present historical validation as a fresh result.

Documentation-only changes need link/example validation and diff review, not artificial behavioral tests or an unnecessary full application test run.

## Tests and checks

Use the existing project environment:

```sh
env/bin/python -m pytest tests/unit -q
env/bin/python -m pytest tests/integration/database -q
env/bin/python -m pytest tests -q
env/bin/python -m compileall -q app tests
git diff --check
```

- Organize tests by domain inside `tests/unit/` and `tests/integration/`: database, embeddings, processing, storage, and operations. Keep fixtures at the narrowest useful scope.
- Unit tests isolate behavior with fakes/mocks; integration tests verify actual boundaries. Follow current fixture usage rather than moving tests based solely on decorators.
- Database integrations require `RYPIQ_TEST_ADMIN_URL` pointing to a disposable PostgreSQL server with pgvector. Use unique temporary databases and deterministic cleanup. Never substitute the application's configured database.
- Model integrations require a local model; do not download one merely to inspect or validate unrelated work.
- Skipped integrations are unverified, not passing evidence. Report unavailable prerequisites clearly.
- Run configured checks appropriate to the change. Do not invent successful lint/type-check results when no such checks were run.

## Error and resource rules

Validate inputs before side effects where possible. Preserve expected exceptions and return contracts. CLI boundaries translate errors into nonzero exits and safe diagnostics; lower layers must not print or terminate the process.

Use parameterized SQL and safe identifier handling. Never log credentials, full connection URLs, or raw driver diagnostics that may contain sensitive values. Keep secrets out of examples, fixtures, and reports.

The creator of a resource owns its cleanup. Prefer `with create_context() as context:` or `with get_database() as database:` at composition boundaries. Cover cleanup on both success and failure. Keep database transaction boundaries explicit; multi-system workflows need documented compensation because filesystem and database changes are not one atomic transaction.
