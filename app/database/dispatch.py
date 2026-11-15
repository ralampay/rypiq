def dispatch(args, service):
    methods = {
        "db:create": "create",
        "db:drop": "drop",
        "db:upgrade": "generate",
        "db:migrate": "migrate",
        "db:rollback": "rollback",
        "db:status": "status",
        "db:migration": "new_migration",
    }

    try:
        method = methods[args.command]
    except KeyError:
        raise ValueError(
            f"Unknown database command {args.command}"
        ) from None

    arguments = (args.name,) if args.command == "db:migration" else ()

    return getattr(service, method)(*arguments)
