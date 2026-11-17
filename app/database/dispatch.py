"""Each database command has one explicit purpose."""
METHODS = {
    'db:create': 'create',
    'db:migrate': 'migrate',
    'db:drop': 'drop',
    'db:reset': 'reset',
}


def dispatch(args, service):
    try:
        method = METHODS[args.command]
    except KeyError:
        raise ValueError(f'Unknown database command {args.command}') from None
    return getattr(service, method)()
