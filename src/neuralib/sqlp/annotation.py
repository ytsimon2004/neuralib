
import typing
from typing import TYPE_CHECKING

from .literal import FOREIGN_POLICY, CONFLICT_POLICY

if TYPE_CHECKING:
    pass

__all__ = [
    'CURRENT_DATE', 'CURRENT_TIME', 'CURRENT_TIMESTAMP',
    'PRIMARY', 'UNIQUE', 'unique', 'foreign', 'check',
]

T = typing.TypeVar('T')
missing = object()


@typing.final
class PRIMARY(object):
    """
    annotate a field as a primary key.

    >>> class Example:
    ...     a: Annotated[str, PRIMARY]  # style 1
    """

    def __init__(self, order: typing.Literal['ASC', 'DESC'] = None,
                 conflict: CONFLICT_POLICY = None,
                 auto_increment=False):
        self.order = order
        self.conflict = conflict
        self.auto_increment = auto_increment


@typing.final
class UNIQUE(object):
    """
    annotated a field as a unique key.

    >>> class Example:
    ...     a: Annotated[str, UNIQUE]  # style 1
    """

    def __init__(self, conflict: CONFLICT_POLICY = None):
        self.conflict = conflict


@typing.final
class CURRENT_DATE(object):
    """
    annotated a date field which use current date as its default.

    >>> class Example:
    ...     a: Annotated[datetime.date, CURRENT_DATE]

    """

    def __init__(self):
        raise RuntimeError()


@typing.final
class CURRENT_TIME(object):
    """
    annotated a time field which use current time as its default.

    >>> class Example:
    ...     a: Annotated[datetime.time, CURRENT_TIME]

    """

    def __init__(self):
        raise RuntimeError()


@typing.final
class CURRENT_TIMESTAMP(object):
    """
    annotated a datetime field which use current datetime as its default.

    >>> class Example:
    ...     a: Annotated[datetime.datetime, CURRENT_TIMESTAMP]

    """

    def __init__(self):
        raise RuntimeError()


def unique(conflict: CONFLICT_POLICY = None):
    """
    A decorator to create an unique constraint.
    """

    def _decorator(f):
        setattr(f, '_sql_unique', (conflict,))
        return f

    return _decorator


def foreign(*field,
            update: FOREIGN_POLICY = 'NO ACTION',
            delete: FOREIGN_POLICY = 'NO ACTION'):
    """
    A decorator to create a foreign constraint.

    Common use:

    With a foreign table

    >>> class ForeignTable:
    ...     a: Annotated[str, str]
    ...     b: Annotated[str, str]

    1. mapping one-by-one

    >>> class Example:
    ...     a: Annotated[str, str]
    ...     b: Annotated[str, str]
    ...     @foreign(ForeignTable.a, ForeignTable.b)
    ...     def _foreign(self):
    ...         return self.a, self.b

    2. by default, with the primary keys for the referred foreign table.

    >>> class Example:
    ...     a: Annotated[str, str]
    ...     b: Annotated[str, str]
    ...     @foreign(ForeignTable)
    ...     def _foreign(self):
    ...         return self.a, self.b

    3. Self refered.

    >>> class Example:
    ...     a: Annotated[str, str]
    ...     b: Annotated[str, str]
    ...     @foreign('a')
    ...     def _foreign(self):
    ...         return self.b

    :param field: a foreign table or foreign fields.
    :param update: update policy
    :param delete: delete policy
    """
    if len(field) == 0:
        raise RuntimeError('empty fields')

    def _decorator(f):
        setattr(f, '_sql_foreign', (field, update, delete))
        return f

    return _decorator


def check(field: str = None):
    """
    A decorator to make a check constraint.

    1. by a field

    >>> class Example:
    ...     a: str
    ...     @check('a')
    ...     def check_a(self):
    ...         return self.a != ''

    2. by over all

    >>> class Example:
    ...     a: str
    ...     @check()
    ...     def check_a(self):
    ...         return self.a != ''

    :param field: field name.
    """

    def _decorator(f):
        setattr(f, '_sql_check', (field, ))
        return f

    return _decorator
