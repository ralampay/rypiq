import importlib
import inspect
import pkgutil
import types
from dataclasses import fields, is_dataclass
from datetime import datetime
from typing import Union, get_args, get_origin, get_type_hints
from uuid import UUID

def persistent(table_name, *, primary_key="id"):
    def decorate(model):
        model.__persistence__ = (table_name, primary_key)

        return model

    return decorate

def discover(package_name="app.models"):
    importlib.invalidate_caches()
    package = importlib.import_module(package_name)
    names = [package_name]
    names.extend(
        item.name
        for item in pkgutil.walk_packages(
            package.__path__, package.__name__ + "."
        )
    )

    found = {}
    for name in sorted(names):
        module = importlib.import_module(name)
        for _, model in inspect.getmembers(module, inspect.isclass):
            if (
                model.__module__ == module.__name__
                and "__persistence__" in vars(model)
            ):
                found[(model.__module__, model.__qualname__)] = model

    return [found[key] for key in sorted(found)]

def metadata_for(models):
    # Lazy imports keep Chroma model discovery independent of SQLAlchemy
    import sqlalchemy as sa
    from sqlalchemy.dialects import postgresql as pg

    metadata = sa.MetaData(naming_convention={
        "pk": "pk_%(table_name)s",
        "ix": "ix_%(table_name)s_%(column_0_name)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s"
    })

    scalar = {
        str: sa.Text,
        int: sa.Integer,
        float: pg.DOUBLE_PRECISION,
        bool: sa.Boolean,
        UUID: lambda: pg.UUID(as_uuid=True),
        datetime: lambda: sa.DateTime(timezone=True),
        dict: pg.JSONB,
    }

    for model in models:
        if not is_dataclass(model):
            raise ValueError(f"{model.__name__} must be a dataclass")

        name, primary_key = model.__persistence__
        
        if name in metadata.tables:
            raise ValueError(f"duplicate persistent table: {name}")

        hints = get_type_hints(model)

        columns = []

        for item in fields(model):
            annotation = hints[item.name]
            nullable = False

            if get_origin(annotation) in (Union, types.UnionType):
                arguments = get_args(annotation)
                nullable = type(None) in arguments
                remaining = [a for a in arguments if a is not type(None)]

                if not nullable or len(remaining) != 1:
                    raise ValueError(
                        f"Unsupported {model.__name__}.{item.name}"
                    )
                        
                annotation = remaining[0]

            if (
                get_origin(annotation) is list
                and get_args(annotation) == (float,)
            ):
                sql_type = pg.ARRAY(pg.DOUBLE_PRECISION())
            elif annotation in scalar:
                sql_type = scalar[annotation]()
            else:
                raise ValueError(
                    f"Unsupported {model.__name__}.{item.name}"
                )

            is_key = item.name == primary_key

            if is_key and nullable:
                raise ValueError("Primary key cannot be nullable")
            columns.append(sa.Column(
                item.name, sql_type,
                primary_key=is_key,
                nullable=nullable and not is_key,
            ))

        if primary_key not in {column.name for column in columns}:
            raise ValueError(f"Missing primary key: {name}.{primary_key}")
        sa.Table(name, metadata, *columns)

    return metadata
