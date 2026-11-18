"""Console entrypoint: parse, dispatch, serialize, and report safe errors."""
from collections.abc import Sequence
import json

from sqlalchemy.exc import SQLAlchemyError

from app.cli_parser import build_parser
from app.diagnostics import ConfigurationReport, check_configuration
from app.cli_handlers import execute
from app.cli_output import embedding_preview, validate_export_path, write_json_export


def format_application_table(records: list[dict]) -> str:
    lines = [f'{"ID":<36}  NAME']
    for record in records:
        name = ''.join(character if character.isprintable() else json.dumps(character)[1:-1]
                       for character in record['name'])
        lines.append(f'{record["id"]:<36}  {name}')
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


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.doctor and args.command is not None:
        parser.error('--doctor cannot be combined with a subcommand')
    if not args.doctor and args.command is None:
        parser.error('the following arguments are required: command')
    try:
        if args.doctor:
            report = check_configuration()
            print(format_doctor_report(report))
            if report.incomplete_count:
                raise SystemExit(1)
            return
        embedding_output = args.command in ('embeddings:create', 'embeddings:show')
        destination = None
        if embedding_output and args.output:
            destination = validate_export_path(args.output, source=args.path if args.command == 'embeddings:create' else None)
        result = execute(args)
        if destination is not None:
            try:
                write_json_export(destination, result)
            except (OSError, ValueError, TypeError):
                if args.command == 'embeddings:create':
                    ids = ', '.join(chunk['id'] for chunk in result['chunks'])
                    raise RuntimeError(f'Embeddings persisted but JSON export failed. Chunk IDs: {ids}. '
                                       'Retry export with embeddings:show --id ID --output PATH.') from None
                raise RuntimeError('Embedding loaded but JSON export failed; check the destination.') from None
        if embedding_output and not args.embeddings_full:
            result = embedding_preview(result)
        if args.command == 'applications:list' and args.output == 'text':
            serialized = format_application_table(result)
        else:
            serialized = json.dumps(result, indent=2, allow_nan=False)
        print(serialized)
    except SQLAlchemyError:
        parser.error('Database operation failed; check connectivity, permissions, pgvector availability and schema configuration')
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        parser.error(str(exc))
    except Exception:
        # Unexpected driver/plugin diagnostics can contain credentials.
        parser.error('Command failed; check configuration and dependencies')


if __name__ == '__main__':
    main()
