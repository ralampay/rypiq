# Exceptions for Python callers

Import expected failures from `app.exceptions`. Catch a concrete type when you can
recover, a family when recovery is shared, or `RyPiqError` at your application's
presentation boundary. Successful return values and CLI JSON formats are unchanged.
Exceptions do not carry a universal result envelope or trigger automatic retries.

## Public hierarchy

All the classes below derive from `RyPiqError`. Indentation indicates inheritance;
additional compatibility bases appear in the next table.

```text
RyPiqError
├── ValidationError
│   ├── InvalidIdentifierError
│   ├── InvalidEmbeddingError
│   ├── InvalidStorageKeyError
│   └── ModelMismatchError
├── ConfigurationError
│   ├── DatabaseConfigurationError
│   ├── StorageConfigurationError
│   ├── ProviderConfigurationError
│   └── ModelCatalogError
├── NotFoundError
│   ├── ApplicationNotFoundError
│   ├── DocumentNotFoundError
│   └── EmbeddingNotFoundError
├── ConflictError
│   ├── DuplicateDocumentNameError
│   ├── ApplicationInUseError
│   └── ImmutableFieldError
├── DatabaseError
│   ├── DatabaseConnectionError
│   ├── DatabaseIntegrityError
│   └── SchemaCompatibilityError
├── ProviderError
│   ├── ProviderInitializationError
│   ├── ProviderRequestError
│   ├── ProviderResponseError
│   ├── ProviderCleanupError (also ResourceCleanupError)
│   └── ModelSessionStateError
├── ProcessingError
│   ├── UnsupportedFileTypeError
│   ├── DocumentExtractionError
│   └── EmptyDocumentError
├── StorageError
│   ├── StorageNotFoundError
│   ├── StorageAlreadyExistsError
│   └── StoragePermissionError
├── ResourceCleanupError
└── PartialOperationError
    ├── DatabaseInitializationError
    ├── DatabaseResetError
    ├── StoragePublicationError
    └── ExportError
```

| Family or exception | Compatible built-in catches |
| --- | --- |
| Validation, configuration, conflict, processing | `ValueError` |
| Missing records | `KeyError` |
| Database errors | `RuntimeError` |
| `DatabaseConnectionError`, `SchemaCompatibilityError` | Also `ValueError` |
| Provider errors | `RuntimeError` |
| `ProviderResponseError` | Also `ValueError`, preserving local embedding response catches |
| Storage errors | `OSError` |
| `StorageNotFoundError` | Also `FileNotFoundError` |
| `StorageAlreadyExistsError` | Also `FileExistsError` |
| `StoragePermissionError` | Also `PermissionError` |
| Cleanup and partial-operation errors | `RuntimeError` |

`app.providers.contracts.ProviderError` remains an alias for the exact same class.
Existing `ValueError`, `KeyError`, `RuntimeError`, and storage filesystem catches
continue to work for their corresponding expected failures. New code should prefer
the concrete exceptions. Messages are diagnostics, not machine-readable codes.
Missing-record exceptions retain their prior positional arguments and `KeyError`
string formatting; use `resource_id` instead of parsing those arguments.

Database adapters now translate SQLAlchemy exceptions. Applications that previously
caught SQLAlchemy types around RyPiq calls must catch `DatabaseError` or a subclass
instead. Direct use of a SQLAlchemy engine still raises SQLAlchemy exceptions.
Malformed configuration that previously leaked incidental `TypeError`, `KeyError`,
YAML errors, or file-reading failures now raises the corresponding configuration error.

## Boundaries and recovery

| Failure | Handling guidance |
| --- | --- |
| `ValidationError` | Correct inputs before retrying. Invalid vectors differ from mismatched application/model settings. |
| `ConfigurationError` | Correct the selected configuration, provider options, or catalog. A missing local inference model is a `ProviderConfigurationError`. |
| `NotFoundError` | Inspect `resource_id` (`UUID`) and select an existing record. Missing parents required for writes use the parent's type. |
| `DuplicateDocumentNameError` | Inspect `application_id` and normalized `name`; choose another name. Preflight and the database uniqueness race raise the same type. |
| `ApplicationInUseError` | Existing documents prevent deletion or changes to embedding settings. |
| `ImmutableFieldError` | Preserve immutable ownership, model snapshots, API keys, or creation times. |
| `DatabaseConnectionError` | Check connectivity, credentials, and database existence. |
| `DatabaseIntegrityError` | A constraint other than the recognized document-name constraint rejected the operation. |
| `SchemaCompatibilityError` | Inspect the schema; migration did not accept it. |
| `ProviderInitializationError` | Provider client or local model construction failed. |
| `ProviderRequestError` | The embedding, chat, or discovery request failed. Do not infer retry safety from the type alone. |
| `ProviderResponseError` | Provider output is malformed or incompatible; changing user inputs may not fix it. |
| `ModelSessionStateError` | The session is closed/closing, or closing it would conflict with a held lease. |
| `ProcessingError` | Use a supported, readable document with usable text; malformed, encrypted, and invalid UTF-8 inputs are extraction failures. |
| `StorageError` | Check object existence, destination collisions, permissions, or storage availability. |

Repository `get()` and `details()` methods still return `None` when absent; services
and show operations raise missing-record errors when the record is required.
Deleting an absent record or stored object still returns `False`. Neither case is
converted into an error by this refactor.

Ordinary source-file reads retain native filesystem exceptions. For example,
missing processing/ingestion input and a missing local embedding model still raise
`FileNotFoundError`. Source reads during storage upload also retain native I/O
errors. Stored-object failures use the `StorageError` family instead. Programming
errors and exceptions raised by arbitrary injected dependencies are not universally
wrapped. `RyPiqError` is therefore not a catch-all for every Python failure.

```python
from uuid import UUID
from app.application import show_document
from app.exceptions import DocumentNotFoundError, DatabaseError


def load_document(document_id: UUID):
    try:
        return show_document(document_id=document_id)
    except DocumentNotFoundError as error:
        # Decide how your application represents a missing record.
        assert error.resource_id == document_id
        return None
    except DatabaseError:
        # Let the caller's infrastructure policy handle an unavailable database.
        raise
```

## Partial completion and cleanup

`PartialOperationError` provides keyword-only `operation: str`,
`completed_steps: tuple[str, ...]`, `resource_id: UUID | str | None`, and
`storage_key: str | None`. Identifiers are present when the operation knows them;
they are not full database URLs or provider credentials.

| Exception | `operation` | `completed_steps` | Recovery |
| --- | --- | --- | --- |
| `DatabaseInitializationError` | `db:create` | `('create_database',)` | Database creation succeeded; fix the cause and retry `db:migrate`. |
| `DatabaseResetError` | `db:reset` | `('drop',)` | The drop step completed; recreation failed. Do not assume old data remains. |
| `StoragePublicationError` | `storage:save` | `('publish',)` | `storage_key` identifies the complete published object; an owned staging file remains. |
| `ExportError` | `documents:create` | `('persist',)` | `resource_id` identifies the persisted document. Retry export using `documents:show`, not ingestion. |
| `ExportError` | `documents:show` or `embeddings:show` | `('load',)` | The record loaded, but export failed; retry with a valid destination. |

An export error does not delete database records or originals and does not guarantee
the export destination is absent. A reset error reports the completed drop step,
including an already-absent database; it does not assert that recreation made no
progress. Partial-operation exceptions form their own family: catch that family
alongside `DatabaseError` or `StorageError` when using lifecycle/publication APIs.
The CLI catches `ExportError` and prints its safe recovery message with the existing
nonzero exit behavior.

Ingestion still commits the document and embeddings in one transaction. If that
fails after storing the original, it attempts to delete its newly stored object.
If compensation also fails, the primary exception survives with a sanitized
`__notes__` entry explaining incomplete cleanup. There is no cross-system atomic
transaction and no automatic retry.

Resource owners attempt cleanup on success and failure. When cleanup fails during
another failure (including an interrupt), the original exception is preserved and
receives a safe note. Cleanup alone raises `ResourceCleanupError`, or
`ProviderCleanupError` for providers. Injected resources remain owned by their caller.
Do not retry a completed operation merely because cleanup failed.

Database, provider, and parser diagnostics are suppressed using `raise ... from None`.
Public messages and library-added notes exclude raw SQL, connection URLs, credentials,
and document contents. Python may still retain the original exception in
`__context__`; do not serialize exception internals or driver objects to users.
Native source-file exceptions and caller-created exceptions are not sanitized by
this contract.

## Extending the library

Use the narrowest existing exception whose recovery behavior matches the failure.
Add a subclass only for a distinct recovery case. Keep the module dependency-free,
validate before effects, and translate infrastructure errors in their adapter.
Do not wrap every exception at an operation boundary or replace an existing typed
error with a generic one. Preserve `KeyboardInterrupt` and `SystemExit`.

See [code conventions](CODE_CONVENTIONS.md), [execution paths](EXECUTION_PATHS.md),
and [database contracts](DATABASE.md) for layering and resource ownership.
