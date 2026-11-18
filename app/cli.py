"""Console entrypoint: parse, dispatch, serialize, and report safe errors."""
from collections.abc import Sequence
import json

from sqlalchemy.exc import SQLAlchemyError

from app.cli_parser import build_parser
from app.cli_handlers import execute
from app.cli_output import embedding_preview, validate_export_path, write_json_export


def format_application_table(records: list[dict]) -> str:
    lines = [f'{"ID":<36}  NAME']
    for record in records:
        name = ''.join(character if character.isprintable() else json.dumps(character)[1:-1]
                       for character in record['name'])
        lines.append(f'{record["id"]:<36}  {name}')
    return '\n'.join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
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
