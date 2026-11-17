"""Console entrypoint: parse, dispatch, serialize, and report safe errors."""
from collections.abc import Sequence
import json
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from app.cli_parser import build_parser
from app.cli_handlers import execute


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = execute(args)
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
