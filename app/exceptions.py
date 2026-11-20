"""Public, dependency-free exceptions for expected RyPiq failures.

Messages are safe for presentation. Compatibility bases preserve existing built-in
catches; callers should prefer the specific types documented in docs/EXCEPTIONS.md.
"""
from uuid import UUID


class RyPiqError(Exception):
    """Base for expected application failures, not programming errors."""


class ValidationError(RyPiqError, ValueError):
    """An input violates the operation's contract."""


class InvalidIdentifierError(ValidationError):
    """A record identifier is not a UUID."""


class InvalidEmbeddingError(ValidationError):
    """A vector has invalid dimensions, values, or magnitude."""


class InvalidStorageKeyError(ValidationError):
    """An object key is invalid or escapes the storage root."""


class ModelMismatchError(ValidationError):
    """The model or embedding settings do not match the application."""


class ConfigurationError(RyPiqError, ValueError):
    """Configuration is missing, unreadable, or invalid."""


class DatabaseConfigurationError(ConfigurationError):
    """Invalid database configuration."""


class StorageConfigurationError(ConfigurationError):
    """Invalid object storage configuration."""


class ProviderConfigurationError(ConfigurationError):
    """Invalid model/provider configuration."""


class ModelCatalogError(ConfigurationError):
    """The catalog or requested model alias is invalid."""


class NotFoundError(RyPiqError, KeyError):
    """A required record does not exist within the requested scope."""

    def __init__(self, *args: object, resource_id: UUID | None = None) -> None:
        super().__init__(*args)
        self.resource_id = resource_id


class ApplicationNotFoundError(NotFoundError):
    """The required application was not found."""


class DocumentNotFoundError(NotFoundError):
    """The required document was not found."""


class EmbeddingNotFoundError(NotFoundError):
    """The required embedding was not found."""


class ConflictError(RyPiqError, ValueError):
    """The requested change conflicts with existing state."""


class DuplicateDocumentNameError(ConflictError):
    """A document name is already taken within an application."""

    def __init__(self, message: str = 'A document with this name already exists in the application',
                 *, application_id: UUID | None = None, name: str | None = None) -> None:
        super().__init__(message)
        self.application_id = application_id
        self.name = name


class ApplicationInUseError(ConflictError):
    """Existing documents prevent deleting or reconfiguring an application."""


class ImmutableFieldError(ConflictError):
    """A change would modify immutable ownership or model information."""


class DatabaseError(RyPiqError, RuntimeError):
    """A database operation failed; driver details are suppressed."""


class DatabaseConnectionError(DatabaseError, ValueError):
    """The configured database could not be reached or authenticated."""


class DatabaseIntegrityError(DatabaseError):
    """A database constraint rejected the change."""


class SchemaCompatibilityError(DatabaseError, ValueError):
    """The existing schema cannot be initialized or upgraded safely."""


class ProviderError(RyPiqError, RuntimeError):
    """Safe provider failure; upstream diagnostics are never included."""


class ProviderInitializationError(ProviderError):
    """A provider client or local model could not be initialized."""


class ProviderRequestError(ProviderError):
    """Embedding, inference, or model discovery failed."""


class ProviderResponseError(ProviderError, ValueError):
    """A provider returned malformed or incompatible output."""


class ModelSessionStateError(ProviderError):
    """The model session is closed or has an incompatible active lease."""


class ProcessingError(RyPiqError, ValueError):
    """A document could not be processed."""


class UnsupportedFileTypeError(ProcessingError):
    """No extractor supports the input extension."""


class DocumentExtractionError(ProcessingError):
    """The document is malformed, encrypted, or cannot be decoded."""


class EmptyDocumentError(ProcessingError):
    """Embedding or ingestion requires nonempty extracted text."""


class StorageError(RyPiqError, OSError):
    """Object storage I/O failed."""


class StorageNotFoundError(StorageError, FileNotFoundError):
    """A requested stored object does not exist."""


class StorageAlreadyExistsError(StorageError, FileExistsError):
    """Publishing or copying an object would overwrite an existing file."""


class StoragePermissionError(StorageError, PermissionError):
    """Object storage access was denied."""


class ResourceCleanupError(RyPiqError, RuntimeError):
    """An owned resource could not be released."""


class ProviderCleanupError(ProviderError, ResourceCleanupError):
    """A provider could not be released."""


class PartialOperationError(RyPiqError, RuntimeError):
    """An operation failed after one or more consequential steps completed."""

    def __init__(self, message: str, *, operation: str,
                 completed_steps: tuple[str, ...], resource_id: UUID | str | None = None,
                 storage_key: str | None = None) -> None:
        super().__init__(message)
        self.operation = operation
        self.completed_steps = tuple(completed_steps)
        self.resource_id = resource_id
        self.storage_key = storage_key


class DatabaseInitializationError(PartialOperationError):
    """Database creation succeeded but schema initialization failed."""


class DatabaseResetError(PartialOperationError):
    """The drop step succeeded but database recreation failed."""


class StoragePublicationError(PartialOperationError):
    """The object was published but its staging file could not be removed."""


class ExportError(PartialOperationError):
    """A record was loaded or persisted but JSON export failed."""


class ProcessingBusyError(ConflictError):
    """Another processor currently holds the document lock."""
