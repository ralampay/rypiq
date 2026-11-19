# Logging

RyPiq uses structlog with standard Python logging. Normal CLI commands initialize
logging after loading `.env` and close their handlers when they exit. `--help`,
argument parsing failures, and the read-only `--doctor` command do not initialize
file logging.

Defaults require no configuration: UTF-8, append-only, readable text at INFO level,
no console output, and `logs/<environment>.log` under the project root. Thus the
usual files are `logs/development.log`, `logs/production.log`, and `logs/test.log`.
Timestamps are UTC. Text includes named fields without ANSI colors; JSON uses one
object per line. CLI result JSON stays on stdout.

## Configuration

Environment selection is explicit Python `environment`, then `LOG_ENV`, then
`DATABASE_ENV`, then `development`. Names must start with an ASCII letter or digit
and contain only ASCII letters, digits, underscores, or hyphens.

Optionally copy `config/logging.yml.dist` to `config/logging.yml`. Environment
sections may set `directory`, `level`, `format`, and `console`. Missing sections
use defaults. Unknown settings or invalid values fail initialization.

Settings precedence is explicit Python overrides, environment variables, YAML,
then defaults. `.env` is loaded without replacing existing process variables.

| Variable | Meaning |
| --- | --- |
| `RYPIQ_LOGGING_CONFIG` | Override YAML path; explicitly selected files must exist |
| `RYPIQ_LOG_DIR` | Output directory; relative directories use the project root |
| `RYPIQ_LOG_LEVEL` | DEBUG, INFO, WARNING, ERROR, or CRITICAL; case insensitive |
| `RYPIQ_LOG_FORMAT` | `text` or `json` |
| `RYPIQ_LOG_CONSOLE` | `true`/`1` or `false`/`0`; enables additional stderr output |

An explicit Python `config_path` overrides `RYPIQ_LOGGING_CONFIG`. Relative config
paths use the current working directory. Local YAML and generated logs are ignored
by Git. No secrets belong in this configuration.

## Python and future service startup

```python
from app.logging import configure_logging, get_logger, log_context, shutdown_logging

configure_logging(environment='production', format='json')
try:
    logger = get_logger(__name__)
    with log_context(request_id='request-123'):
        logger.info('document_processed', document_id='document-123', chunk_count=4)
finally:
    shutdown_logging()
```

For fully resolved configuration, pass a `LoggingSettings` instance instead of
keyword overrides. `load_logging_settings()` returns validated settings without
opening log files. `configure_logging()` returns the effective settings.

Initialize once per application process, including each future web worker, and
shut down at process/application exit, not after each request. Repeated identical
configuration is idempotent; changing settings replaces and closes only handlers
created by RyPiq. Failed replacement leaves the existing output active. Imports,
logger retrieval, and business operations do not initialize logging. Without
initialization, library logs are silent unless the host attaches its own handlers.

`get_logger()` uses structlog processors local to RyPiq; it does not change global
structlog configuration. Standard records under `logging.getLogger('rypiq.name')`
use the same output formatting. Root and third-party loggers are not captured or
reconfigured. Hosts should use a distinct child of `rypiq` for their own records.

`log_context()` restores previous tags on exit, including failure and nesting.
Context variables isolate concurrent asynchronous tasks. Thread boundaries need
explicit context propagation; no web middleware is included.

## Events and failures

CLI, ingestion, embedding, search, and chat emit `operation_started`,
`operation_completed`, and `operation_failed`. Completion/failure records include
`elapsed_ms`. IDs and counts are included where available. Nested workflow events
retain CLI context. Failures log exception types and re-raise the original error.
No automatic event includes raw driver diagnostics, exception text/tracebacks,
document contents, query text, prompts, vectors, credentials, or connection URLs.
Callers must likewise pass safe fields to their logger; arbitrary message text is
not automatically scrubbed.

Invalid configuration or inability to open the file raises a safe
`ConfigurationError` before the command runs. Later file write failures emit a
fixed stderr warning without exposing diagnostics or changing business results.

## Rotation and validation

Use external rotation (for example, logrotate). On Linux, the file handler detects
a replaced log file and reopens it. There is no built-in size rotation or retention;
configure those in your deployment. Multiple processes can append, but this is not
a coordinated logging service and does not guarantee ordering across processes.

Unit tests use temporary directories and fake providers; database/model access is
not required to validate logging. See [project setup](../README.md).
