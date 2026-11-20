"""Console entrypoint: parse, dispatch, serialize, and report safe errors."""
from collections.abc import Sequence
import json
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from app.config import AppConfig, load_config
from app.logging import configure_logging, log_operation, shutdown_logging

from app.exceptions import RyPiqError, ExportError, ResourceCleanupError, ValidationError
from app.cli_parser import build_parser
from app.diagnostics import ConfigurationReport, check_configuration
from app.cli_handlers import execute
from app.cli_output import embedding_preview, validate_export_path, write_json_export


def format_application_table(records: list[dict]) -> str:
    columns = [('id', 'ID'), ('name', 'NAME'), ('embedding_provision_type', 'EMBEDDING PROVIDER'),
               ('inference_provision_type', 'INFERENCE PROVIDER'),
               ('embedding_model', 'EMBEDDING MODEL'), ('embedding_size', 'EMBEDDING SIZE'),
               ('inference_model', 'INFERENCE MODEL'), ('document_count', 'DOCUMENTS')]
    rows = [[heading for _, heading in columns]]
    for record in records:
        row = []
        for key, _ in columns:
            value = record[key]
            if key in ('embedding_model', 'inference_model') and not value:
                value = '-'
            row.append(''.join(char if char.isprintable() else json.dumps(char)[1:-1]
                               for char in str(value)))
        rows.append(row)
    widths = [max(len(row[index]) for row in rows) for index in range(len(columns))]
    return '\n'.join('  '.join(cell.ljust(width) if index < len(columns) - 1 else cell
                               for index, (cell, width) in enumerate(zip(row, widths, strict=True)))
                     for row in rows)


def format_document_table(records: list[dict]) -> str:
    names = [''.join(char if char.isprintable() else json.dumps(char)[1:-1]
                     for char in record['name']) for record in records]
    width = max([len('NAME'), *map(len, names)])
    lines = [f'{"ID":<36}  {"NAME":<{width}}  STATUS      EMBEDDINGS']
    lines.extend(f'{record["id"]:<36}  {name:<{width}}  {record["status"]:<10}  {record["embedding_count"]}'
                 for record, name in zip(records, names, strict=True))
    return '\n'.join(lines)


def format_doctor_report(report: ConfigurationReport) -> str:
    def one_line(value: str) -> str:
        return ''.join(char if char.isprintable() else json.dumps(char)[1:-1] for char in value)

    lines = [f'Configuration check — {one_line(report.environment)}', '']
    lines.extend(f'{check.status:<9} {one_line(check.name)} — {one_line(check.detail)}'
                 for check in report.checks)
    lines.append('')
    if report.incomplete_count:
        lines.append(f'Configuration incomplete: {report.incomplete_count} required item(s) need attention.')
    else:
        lines.append('Required configuration and database checks passed. Optional items may still need attention.')
    lines.append('Saved applications were not checked.')
    lines.append('Application model, provider, and chunking settings were not verified. No models were loaded.')
    lines.append('Filesystem permission checks do not guarantee a later write will succeed.')
    return '\n'.join(lines)


def processing_failure_context(failure: Exception) -> str:
    identifier = getattr(failure, '_rypiq_document_id', None)
    if not isinstance(identifier, UUID):
        return ''
    message = f' Document ID: {identifier}. Retry with documents:process --document-id {identifier}.'
    status_failure = 'Could not record error status; the document may still be pending or processing.'
    if status_failure in getattr(failure, '__notes__', ()):
        message += ' ' + status_failure
    return message


def main(argv: Sequence[str] | None = None, *, config: AppConfig | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.doctor and args.command is not None:
        parser.error('--doctor cannot be combined with a subcommand')
    if not args.doctor and args.command is None:
        parser.error('the following arguments are required: command')
    logging_initialized = False
    try:
        if args.command != 'server':
            config = load_config() if config is None else config
        if args.doctor:
            report = check_configuration(config=config)
            print(format_doctor_report(report))
            if report.incomplete_count:
                raise SystemExit(1)
            return
        if args.command == 'server':
            import uvicorn
            if config is not None:
                if args.reload:
                    raise ValidationError('Injected server configuration cannot be combined with --reload')
                from app import create_app
                uvicorn.run(create_app(config=config), host=args.host, port=args.port)
            else:
                uvicorn.run('app:create_app', factory=True, host=args.host, port=args.port, reload=args.reload)
            return
        configure_logging(settings=config.require_logging())
        logging_initialized = True
        if args.command == 'worker':
            from app.worker import run_worker
            run_worker(once=args.once, config=config)
            return
        with log_operation('cli', command=args.command):
            exportable_output = args.command in ('documents:create', 'documents:process', 'documents:show', 'embeddings:show')
            destination = None
            if exportable_output and args.output:
                destination = validate_export_path(args.output, source=args.path if args.command == 'documents:create' else None)
            result = execute(args, config=config)
            if destination is not None:
                try:
                    write_json_export(destination, result)
                except (OSError, ValueError, TypeError, ResourceCleanupError):
                    if args.command in ('documents:create', 'documents:process'):
                        raise ExportError(f"Document persisted but JSON export failed. Document ID: {result['id']}. "
                                           'Retry export with documents:show --id ID --output PATH.',
                                          operation=args.command, completed_steps=('persist',), resource_id=result['id']) from None
                    raise ExportError('Record loaded but JSON export failed; check the destination.',
                                      operation=args.command, completed_steps=('load',), resource_id=result['id']) from None
            if args.command == 'embeddings:show' and not args.embeddings_full:
                result = embedding_preview(result)
            if args.command == 'applications:list' and args.output == 'text':
                serialized = format_application_table(result)
            elif args.command == 'documents:list' and args.output == 'text':
                serialized = format_document_table(result)
            else:
                serialized = json.dumps(result, indent=2, allow_nan=False)
            print(serialized)
    except RyPiqError as exc:
        parser.error(str(exc) + processing_failure_context(exc))
    except SQLAlchemyError as exc:
        parser.error('Database operation failed; check connectivity, permissions, pgvector availability and schema configuration' + processing_failure_context(exc))
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.error(str(exc) + processing_failure_context(exc))
    except Exception as exc:
        # Unexpected driver/plugin diagnostics can contain credentials.
        parser.error('Command failed; check configuration and dependencies' + processing_failure_context(exc))
    finally:
        if logging_initialized:
            shutdown_logging()


if __name__ == '__main__':
    main()
