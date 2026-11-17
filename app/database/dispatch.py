"""Each database command has one explicit purpose."""
METHODS = {
    'db:create': 'create',
    'db:migrate': 'migrate',
    'db:drop': 'drop',
    'db:reset': 'reset',
}


def method_for(command: str) -> str:
    try:
        return METHODS[command]
    except KeyError:
        raise ValueError(f'Unknown database command {command}') from None


def dispatch(args, service):
    """Compatibility adapter for callers holding a parsed namespace."""
    return getattr(service, method_for(args.command))()
