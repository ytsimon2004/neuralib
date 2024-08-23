import contextvars
import sqlite3
from pathlib import Path
from typing import Union, TypeVar, Optional, Any

import polars as pl

from .literal import UPDATE_POLICY
from .stat import SqlStat
from .table import table_name, table_field_names

__all__ = ['Connection', 'get_connection_context']

T = TypeVar('T')
S = TypeVar('S')


class Connection:
    """
    A sqlite3 connection wrapper.

    If as a context manager that put itself in a global context-aware variable.
    """

    def __init__(self, filename: Union[str, Path] = ':memory:', *,
                 debug: bool = False):
        """
        :param filename: sqlite database filepath. use in-memory database by default.
        :param debug: print statement when executing.
        """
        self._connection = sqlite3.Connection(str(filename))
        self._debug = debug
        self._context: Optional[contextvars.Token] = None

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection

    def __enter__(self):
        self._context = CONNECTION.set(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is None:
            self._connection.commit()
        else:
            self._connection.rollback()

        if (token := self._context) is not None:
            CONNECTION.reset(token)
            self._context = None

    def __del__(self):
        self._connection.close()

    # ====== #
    # tables #
    # ====== #

    def list_table(self) -> list[str]:
        """
        list table's name stored in the database.

        :return: list of table's name
        """
        results = self.execute('SELECT name FROM sqlite_master WHERE type = "table"')
        return [it[0] for it in results]

    def table_schema(self, table: Union[str, type]) -> str:
        """
        get the table schema stored in the database.

        :param table: table name or type.
        :return: table schema
        """
        if isinstance(table, str):
            name = table
        else:
            name = table_name(table)

        if not isinstance(name, str):
            raise TypeError()
        results = self.execute('SELECT sql FROM sqlite_master WHERE type = "table" AND name = ?', [name]).fetchone()
        if results is None:
            raise ValueError(f'table {table} not found')
        return results[0]

    # ======= #
    # execute #
    # ======= #

    def commit(self):
        self._connection.commit()

    def rollback(self):
        self._connection.rollback()

    def execute(self, stat: Union[str, SqlStat], parameter: Union[list[Any], dict[str, Any]] = ()) -> sqlite3.Cursor:
        """
        execute a statement.

        :param stat: a raw SQL statement or a SqlStat
        :param parameter: statement variable's value.
        :return: a cursor.
        """
        if isinstance(stat, SqlStat):
            stat._connection = None
            stat, _parameter = stat.build()
            if len(parameter) == 0:
                parameter = _parameter

        if self._debug:
            print(repr(stat))

        try:
            ret = self._connection.execute(stat, parameter)
        except sqlite3.OperationalError as e:
            raise RuntimeError(stat) from e
        except sqlite3.ProgrammingError as e:
            raise RuntimeError(stat) from e
        except sqlite3.InterfaceError as e:
            raise RuntimeError(repr(parameter)) from e

        return ret

    def execute_batch(self, stat: Union[str, SqlStat], parameter: list) -> sqlite3.Cursor:
        """
        execute a statement in batch mode.

        :param stat: a raw SQL statement or a SqlStat
        :param parameter: list of statement variable's value.
        :return: a cursor.
        """
        if isinstance(stat, SqlStat):
            stat._connection = None
            stat, _ = stat.build()

        if self._debug:
            print(repr(stat))

        try:
            ret = self._connection.executemany(stat, parameter)
        except sqlite3.OperationalError as e:
            raise RuntimeError(stat) from e
        except sqlite3.ProgrammingError as e:
            raise RuntimeError(f'{stat=}, ?={parameter}') from e
        except sqlite3.InterfaceError as e:
            raise RuntimeError(repr(parameter)) from e

        return ret

    def execute_script(self, stat: Union[str, list[str], list[SqlStat]]):
        """
        execute SQL script.

        :param stat: a raw SQL script, or a list of statements/SqlStat.
        :param commit: commit script.
        :return: a cursor.
        """
        script = []
        if isinstance(stat, str):
            script.append(stat)
        else:
            for _stat in stat:
                if isinstance(_stat, str):
                    script.append(_stat)
                elif isinstance(_stat, SqlStat):
                    _stat._connection = None
                    script.append(_stat.build()[0])
                else:
                    raise TypeError()

        script = ';\n'.join(script)
        if self._debug:
            print(repr(stat))

        try:
            ret = self._connection.executescript(script)
        except sqlite3.OperationalError as e:
            raise RuntimeError(script) from e
        except sqlite3.ProgrammingError as e:
            raise RuntimeError(script) from e
        except sqlite3.InterfaceError as e:
            raise RuntimeError(script) from e

        return ret

    # ========= #
    # functions #
    # ========= #

    def sqlite_compileoption_get(self, n):
        """
        * https://www.sqlite.org/lang_corefunc.html#sqlite_compileoption_get
        * https://www.sqlite.org/compile.html#omitfeatures

        :param n:
        :return:
        """
        ret, *_ = self._connection.execute("""\
        WITH RET(val) AS (
            VALUES (sqlite_compileoption_get(?))
        )
        SELECT * FROM RET
        """, (n,)).fetchone()
        return ret

    def sqlite_compileoption_used(self, n) -> bool:
        """
        * https://www.sqlite.org/lang_corefunc.html#sqlite_compileoption_used
        * https://www.sqlite.org/compile.html#omitfeatures

        :param n:
        :return:
        """
        ret, *_ = self._connection.execute("""\
        WITH RET(val) AS (
            VALUES (sqlite_compileoption_used(?))
        )
        SELECT * FROM RET
        """, (n,)).fetchone()
        return ret > 0

    # ============= #
    # import/export #
    # ============= #

    def export_dataframe(self, table: Union[str, type[T]]) -> pl.DataFrame:
        """
        export a table into a DataFrame.

        :param table: table name or type.
        :return: polars DataFrame
        """
        if isinstance(table, str):
            from .util import get_fields_from_schema
            fields = get_fields_from_schema(self.table_schema(table))
        else:
            fields = table_field_names(table)
            table = table_name(table)

        result = self.connection.execute(f'SELECT * FROM {table}').fetchall()
        return pl.DataFrame(result, schema=fields)

    def import_dataframe(self, table: Union[str, type[T]], df: pl.DataFrame, *,
                         policy: UPDATE_POLICY = 'REPLACE'):
        """
        Import a table from a DataFrame.

        :param table: table name or type.
        :param df: polars DataFrame
        :param policy: insert policy
        """
        if isinstance(table, str):
            from .util import get_fields_from_schema
            fields = get_fields_from_schema(self.table_schema(table))
        else:
            fields = table_field_names(table)
            table = table_name(table)

        stat = f'INSERT OR {policy.upper()} INTO {table} VALUES (' + ','.join(['?'] * len(fields)) + ')'
        self.execute_batch(stat, [
            tuple([row[f] for f in fields])
            for row in df.iter_rows(named=True)
        ])

    def export_csv(self, table: Union[str, type[T]], file: Union[str, Path]):
        """
        export a table into a csv file.

        :param table: table name or type.
        :param file: csv filepath.
        """
        self.export_dataframe(table).write_csv(file)

    def import_csv(self, table: Union[str, type[T]], file: Union[str, Path]):
        """
        Import a table from a csv file.

        :param table: table name or type.
        :param file: csv filepath
        """
        self.import_dataframe(table, pl.read_csv(file))


CONNECTION = contextvars.ContextVar('CONNECTION', default=None)


def get_connection_context() -> Optional[Connection]:
    """
    Get a connection under the current context.
    """
    return CONNECTION.get()
