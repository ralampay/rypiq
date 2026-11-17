"""Input validation shared by the two CRUD services."""
from uuid import UUID


def validate_id(value: UUID) -> None:
    if not isinstance(value, UUID):
        raise ValueError('ID must be a UUID')


def validate_pagination(limit: int, offset: int) -> None:
    if type(limit) is not int or limit < 1:
        raise ValueError('Limit must be a positive integer')
    if type(offset) is not int or offset < 0:
        raise ValueError('Offset must be a nonnegative integer')
