from collections.abc import Callable
from typing import overload, Any, TypeVar, Union

from .connection import get_connection_context
from .expr import SqlField
from .table import table_name, Field, table_field

__all__ = [
    'rename_table',
    'drop_table',
    'rename_column',
    'add_column',
    'drop_column',
    'migrate_table',
]

T = TypeVar('T')
S = TypeVar('S')


def rename_table(old_table: Union[str, type[T]], new_table: Union[str, type[S]]):
    """
    https://www.sqlite.org/lang_altertable.html#alter_table_rename

    :param old_table:
    :param new_table:
    :return:
    """

    old_table = old_table if isinstance(old_table, str) else table_name(old_table)
    new_table = new_table if isinstance(new_table, str) else table_name(new_table)
    get_connection_context().execute(
        f'ALTER TABLE {old_table} RENAME TO {new_table}',
        commit=True
    )


def drop_table(table: type[T]):
    get_connection_context().execute(
        f'DROP TABLE {table_name(table)}',
        commit=True
    )


def rename_column(table: type[T], old_col: str, new_col: str):
    """
    https://www.sqlite.org/lang_altertable.html#alter_table_rename_column

    :param table:
    :param old_col:
    :param new_col:
    """
    get_connection_context().execute(
        f'ALTER TABLE {table_name(table)} RENAME COLUMN ? TO ?',
        parameter=(old_col, new_col),
        commit=True
    )


@overload
def add_column(table: type[T], col: str):
    pass


@overload
def add_column(table: Union[Field, SqlField, Any]):
    pass


def add_column(table: type[T], col=None):
    """
    https://www.sqlite.org/lang_altertable.html#alter_table_add_column

    :param table:
    :param col:
    :return:
    """
    if isinstance(table, type) and isinstance(col, str):
        field = table_field(table, col)
    elif isinstance(field := table, Field) and col is None:
        table = field.table
    elif isinstance(field := table, SqlField) and col is None:
        table = field.table
    else:
        raise TypeError()

    from .stat import _column_def, SqlStat
    stat = SqlStat(table)
    stat.add(['ALTER', 'TABLE', table_name(table), 'ADD', 'COLUMN'])
    _column_def(stat, field)
    stat.submit(commit=True)


@overload
def drop_column(table: type[T], col: str):
    pass


@overload
def drop_column(table: Union[Field, SqlField, Any]):
    pass


def drop_column(table: type[T], col=None):
    """
    https://www.sqlite.org/lang_altertable.html#alter_table_drop_column

    :param table:
    :param col:
    :return:
    """
    if isinstance(table, type) and isinstance(col, str):
        pass
    elif isinstance(field := table, Field) and col is None:
        table = field.table
        field = field.name
    elif isinstance(field := table, SqlField) and col is None:
        table = field.table
        field = field.name
    else:
        raise TypeError()

    get_connection_context().execute(
        'ALTER TABLE ? DROP COLUMN ?',
        parameter=(table_name(table), field),
        commit=True
    )


def migrate_table(old_table: type[T], new_table: type[S], migrate: Callable[[T], S]):
    """

    https://www.sqlitetutorial.net/sqlite-alter-table/

    :param old_table:
    :param new_table:
    :param migrate:
    :return:
    """
    from . import stat
    if (connection := get_connection_context()) is None:
        raise RuntimeError('not in a connection context')

    connection.execute('PRAGMA foreign_keys=off')

    try:
        stat.create_table(new_table)
        stat.insert_into(new_table).submit(list(map(migrate, stat.select_from(old_table))))
        drop_table(old_table)
    finally:
        connection.execute('PRAGMA foreign_keys=on')
