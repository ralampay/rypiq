"""Console entrypoint: parse, dispatch, serialize, and report safe errors."""
from collections.abc import Sequence
import json
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from app.cli_parser import build_parser
from app.cli_handlers import execute


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
        result = execute(args)
        if args.command == 'applications:list' and args.output == 'text':
            serialized = format_application_table(result)
        else:
            serialized = json.dumps(result, indent=2, allow_nan=False)
        if args.command == 'embed' and args.output:
            Path(args.output).write_text(serialized + '\n', encoding='utf-8')
        else:
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
