from __future__ import annotations

import abc
import functools
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional, Literal, TYPE_CHECKING

import polars as pl

from argclz import AbstractParser, argument

if TYPE_CHECKING:
    from .connection import Connection

__all__ = ['Database', 'transaction']


def transaction():
    """
    A decorator that decorate a function as a transaction session
    that open a connection and set to the global variable.

    >>> @transaction
    ... def example(self):
    ...     self.do_something()

    equalivent to

    >>> def example(self):
    ...     with self.open_connection():
    ...         self.do_something()

    """

    def _decorator(f):
        @functools.wraps(f)
        def _transaction(self: Database, *args, **kwargs):
            with self.open_connection():
                return f(self, *args, **kwargs)

        return _transaction

    return _decorator


class Database(metaclass=abc.ABCMeta):
    """
    A class for the common usage of using sqlite.
    """

    sqlp_debug_mode = False

    @property
    @abc.abstractmethod
    def database_file(self) -> Optional[Path]:
        """sqlite database filepath"""
        pass

    @property
    @abc.abstractmethod
    def database_tables(self) -> list[type]:
        """supporting tables"""
        pass

    def open_connection(self) -> Connection:
        """
        open a connection to the database.
        """
        if (database_file := self.database_file) is None:
            database_file = ':memory:'
        else:
            database_file.parent.mkdir(exist_ok=True)

        from .connection import Connection
        ret = Connection(database_file, debug=self.sqlp_debug_mode)
        cls = type(self)
        if getattr(cls, '__first_connect_init', True):
            from .stat_start import create_table
            with ret:
                for table in self.database_tables:
                    create_table(table).submit()

            setattr(cls, '__first_connect_init', False)

        return ret


class CliDatabase(Database, AbstractParser):
    """
    Wrap Database to support commandline interface.
    """

    sqlp_debug_mode: bool = argument('--debug')

    DB_FILE: str = argument('-d', '--database', metavar='FILE', default=None)
    DB_STAT: list[str] = argument(metavar='STATS', nargs='*', action='extend')
    list_table: str = argument('--table', metavar='NAME', default=None, const='', nargs='?')
    from_file: bool = argument('-f', '--file')
    action: Literal['import', 'export'] = argument('--action', default=None)
    pretty: bool = argument('-p', '--pretty', help='as polars dataframe')
    print_all: bool = argument('-a', '--all', help='print all dataframe if --pretty')

    USAGE = """\
%(prog)s -d FILE --table [NAME]
%(prog)s -d FILE STAT ...
%(prog)s -d FILE --file SCRIPT ...
%(prog)s -d FILE --action=(import|export) --table NAME FILE
"""

    def __init__(self, database: Optional[Database] = None):
        Database.__init__(self)
        self.database = database

    @property
    def database_file(self) -> Optional[Path]:
        if self.database is not None:
            return self.database.database_file
        if self.DB_FILE is None:
            return None
        return Path(self.DB_FILE)

    @property
    def database_tables(self) -> list[type]:
        if self.database is not None:
            return self.database.database_tables
        return []

    def open_connection(self) -> Connection:
        if self.database is not None:
            self.database.sqlp_debug_mode = self.sqlp_debug_mode
            return self.database.open_connection()
        return super().open_connection()

    def run(self):
        if self.action is not None:
            return self.run_action()

        if self.list_table is not None:
            return self.run_list_table()

        if self.from_file:
            return self.run_script()

        return self.run_statement()

    def run_list_table(self):
        with self.open_connection() as connection:
            if len(self.list_table) == 0:
                print(connection.list_table())
            else:
                print(connection.table_schema(self.list_table))

    def run_action(self):
        if self.action == 'export':
            if self.list_table is None or len(self.list_table) == 0:
                raise ValueError('missing --table')

            if len(self.DB_STAT) == 0:
                out = None
            elif len(self.DB_STAT) == 1:
                out = self.DB_STAT[0]
            else:
                raise RuntimeError(f'too many arguments : {self.DB_STAT[1:]}')

            with self.open_connection() as connection:
                ret = connection.export_dataframe(self.list_table).write_csv(out)
            if out is None:
                print(ret)

        elif self.action == 'import':
            if self.list_table is None or len(self.list_table) == 0:
                raise ValueError('missing --table')

            if len(self.DB_STAT) == 0:
                data = pl.read_csv(sys.stdin)
            elif len(self.DB_STAT) == 1:
                data = pl.read_csv(self.DB_STAT[0])
            else:
                raise RuntimeError(f'too many arguments : {self.DB_STAT[1:]}')

            with self.open_connection() as connection:
                connection.import_dataframe(self.list_table, data)

        else:
            raise ValueError(f'unknown action={self.action}')

    def run_script(self):
        args = list(self.DB_STAT[1:])

        with self.open_connection() as connection:
            stat = []

            with Path(self.DB_STAT[0]).open() as file:
                for line in file:
                    stat.append(line)

                    if line.partition('--')[0].rstrip().endswith(';'):
                        result = self._execute_script(connection, ''.join(stat), args)
                        stat = []
                        self._print_result(connection, result)

            if len(stat):
                result = self._execute_script(connection, ''.join(stat), args)
                self._print_result(connection, result)

            if len(args):
                raise RuntimeError(f'remaining {args=}')

    def _execute_script(self, connection, script: str, args: list[str]) -> sqlite3.Cursor:
        try:
            result = connection.execute(script, parameter=args)
            args.clear()
            return result
        except RuntimeError as e:
            if not isinstance(ex := e.__cause__, sqlite3.ProgrammingError):
                raise

                # sqlite3.ProgrammingError: Incorrect number of bindings supplied. The current statement uses 5, and there are 6 supplied.
            message: str = ex.args[0]
            m = re.search(r'The current statement uses (\d+), and there are (\d+) supplied.', message)
            if m is None:
                raise

            need = int(m.group(1))
            total = int(m.group(2))
            if total != len(args):
                raise

        take = args[:need]
        result = connection.execute(script, parameter=take)
        del args[:need]
        return result

    def run_statement(self):
        with self.open_connection() as connection:
            result = connection.execute(' '.join(self.DB_STAT))
            self._print_result(connection, result)

    def _print_result(self, connection, cursor: sqlite3.Cursor):
        if self.pretty:
            from neuralib.sqlp.stat import Cursor
            df = Cursor(connection, cursor).fetch_polars()

            if self.print_all:
                from neuralib.util.verbose import printdf
                printdf(df)
            else:
                print(df)
        else:
            header = cursor.description
            if header is not None:
                print('--', tuple([it[0] for it in header]))
            for data in cursor:
                print(data)


if __name__ == '__main__':
    CliDatabase().main()
