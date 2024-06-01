from __future__ import annotations

import datetime
import sqlite3
import warnings
from collections.abc import Iterator
from typing import overload, TYPE_CHECKING, Any, TypeVar, Generic, Optional, Literal, Union

import polars as pl
from typing_extensions import Self

from .expr import *
from .literal import UPDATE_POLICY, CONFLICT_POLICY
from .table import *

if TYPE_CHECKING:
    from .connection import Connection
    from .table import ForeignConstraint

__all__ = [
    'SqlStat',
    'SqlWhereStat',
    'SqlSelectStat',
    'Cursor',
    'create_table',
    'insert_into',
    'select_from',
    'update',
    'delete_from'
]

T = TypeVar('T')
S = TypeVar('S')


class SqlStat(Generic[T]):
    """Abstract SQL statement."""

    def __init__(self, table: Optional[type[T]]):
        self.table = table
        self._stat = []
        self._para = []
        self._deparameter = False

        from .connection import get_connection_context
        self._connection: Optional[Connection] = get_connection_context()
        # connection will be set to None when following situation happen:
        # * after submit()
        # * __del__() call submit()
        # * as a subquery be used by other statement.
        # TODO ...

    def build(self) -> str:
        """build a SQL statement."""
        ret = ' '.join(self._stat)
        self._stat = None
        return ret

    def submit(self, *, commit=False) -> Cursor[T]:
        """
        build the SQL statement and execute.

        :param commit: commit this statement.
        :return: a cursor
        :raise RuntimeError: current statement does not bind with a connection.
        """
        if (connection := self._connection) is None:
            raise RuntimeError('Do not in a connection context')

        ret = Cursor(connection, connection.execute(self, commit=commit), self.table)
        self._connection = None
        return ret

    def __del__(self):
        # check connection and auto_commit,
        # make SqlStat auto submit itself when nobody is refer to it,
        # so users do need to explict call submit() for every statements.
        if (c := self._connection) is not None and c._auto_commit and self._stat is not None and len(self._stat) > 0:
            try:
                self.submit(commit=True)
            except BaseException as e:
                warnings.warn(repr(e))

    @overload
    def add(self, stat: SqlStat) -> Self:
        pass

    @overload
    def add(self, stat: str | list[str], *para) -> Self:
        pass

    def add(self, stat, *para) -> Self:
        """
        Add SQL token.
        """
        if self._stat is None:
            raise RuntimeError("Statement is closed.")

        if isinstance(stat, SqlStat):
            self._stat.append('(')
            self._stat.extend(stat._stat)
            self._stat.append(')')
            self._para.extend(stat._para)
            stat._connection = None
        elif isinstance(stat, (tuple, list)):
            if self._deparameter:
                para = list(para)
                for _stat in stat:
                    if _stat == '?':
                        self._stat.append(repr(para.pop(0)))
                    else:
                        self._stat.append(_stat)
            else:
                self._stat.extend(stat)
                self._para.extend(para)
        else:
            if self._deparameter:
                if stat == '?':
                    self._stat.append(repr(para[0]))
                else:
                    self._stat.append(stat)
            else:
                self._stat.append(stat)
                self._para.extend(para)

        return self


class Cursor(Generic[T]):
    """
    A SQL cursor wrapper.

    It will try to cast to T from the tuple returns.

    """

    def __init__(self, connection: Connection, cursor: sqlite3.Cursor, table: type[T] | None):
        self._connection = connection
        self._cursor = cursor
        self._table = table
        self.headers: list[str] | None = None
        """headers of return"""

        self._table_cls = None
        if table is not None:
            from .table import _table_class
            try:
                self._table_cls = _table_class(table)
            except AttributeError:
                pass

    def __del__(self):
        self._cursor.close()
        self._connection = None

    def fetchall(self) -> list[T]:
        """fetch all results."""
        return list(self)

    def fetchone(self) -> T | None:
        """fetch the first result."""
        if (ret := self._cursor.fetchone()) is None:
            return None

        if self._table_cls is not None:
            return self._table_cls.table_new(*ret)

        return ret

    def __iter__(self) -> Iterator[T]:
        """iterate the results."""
        table_cls = self._table_cls
        for a in self._cursor:
            if table_cls is not None:
                yield table_cls.table_new(*a)
            else:
                yield a

    def fetch_polars(self) -> pl.DataFrame:
        return pl.DataFrame(list(self._cursor), schema=self.headers)


class SqlWhereStat(SqlStat[T], Generic[T]):
    """statement with **WHERE** support."""

    def where(self, *expr: SqlExpr | Any | None) -> Self:
        """
        ``WHERE`` clause: https://www.sqlite.org/lang_select.html#whereclause

        >>> select_from(A).where( # doctest: SKIP
        ...     A.a == 1, A.b == 2
        ... ).build()
        SELECT * FROM A
        WHERE (A.a = 1) AND (A.b = 2)

        :param expr:
        :return:
        """
        expr = [it for it in expr if it is not None]
        if len(expr):
            from .func_stat import and_
            self.add('WHERE')
            and_(*expr).__sql_stat__(self)
        return self


class SqlSelectStat(SqlWhereStat[T], Generic[T]):
    """**SELECT** statement."""

    def __init__(self, table: type[T] | None, headers: list[str | Field]):
        super().__init__(table)
        self._headers: list[str | Field | None] = headers
        self._window_defs: dict[str, SqlWindowDef] = {}

    def __matmul__(self, other: str) -> SqlAlias[SqlSubQuery]:
        """wrap itself as a subquery with an alias name."""
        return SqlAlias(SqlSubQuery(self), other)

    def fetchall(self) -> list[T]:
        """submit and fetch all result."""
        return self.submit().fetchall()

    def fetchone(self) -> T | None:
        """submit and fetch the first result."""
        return self.submit().fetchone()

    def __iter__(self) -> Iterator[T]:
        """submit and iterate the results"""
        return iter(self.submit())

    def fetch_polars(self) -> pl.DataFrame:
        """submit and fetch all as a polar dataframe."""
        return self.submit().fetch_polars()

    def submit(self, commit=False) -> Cursor[T]:
        ret = super().submit(commit=commit)

        headers = []
        for i, header in enumerate(self._headers):
            if header is None:
                headers.append(f'_{i}')
            elif isinstance(header, str):
                headers.append(header)
            elif isinstance(header, Field):
                name = header.name
                if name not in headers:
                    headers.append(name)
                else:
                    headers.append(f'{header.table_name}.{name}')
            else:
                raise TypeError()
        ret.headers = headers

        return ret

    @overload
    def limit(self, n: int) -> Self:
        pass

    @overload
    def limit(self, row_count: int, offset: int) -> Self:
        pass

    def limit(self, *args: int) -> Self:
        """
        ``LIMIT``: https://www.sqlite.org/lang_select.html#limitoffset

        >>> select_from(A).limit(10).build() # doctest: SKIP
        SELECT * FROM A LIMIT 10
        >>> select_from(A).limit(10, 10).build() # doctest: SKIP
        SELECT * FROM A LIMIT 10 OFFSET 10

        """
        if len(args) == 1:
            n, *_ = args
            self.add(['LIMIT', str(n)])
        elif len(args) == 2:
            row_count, offset = args
            self.add(['LIMIT', str(row_count), 'OFFSET', str(offset)])
        else:
            raise TypeError()
        return self

    def order_by(self, *by: int | str | SqlExpr | Any) -> Self:
        """
        ``ORDER BY``: https://www.sqlite.org/lang_select.html#orderby

        >>> select_from(A).order_by(A.a).build() # doctest: SKIP
        SELECT * FROM A ORDER BY A.

        **possible ordering**

        >>> select_from(A).order_by( # doctest: SKIP
        ...     asc(A.a), desc(A.b), nulls_first(A.c), asc(A.d).nulls_last(),
        ... )
        SELECT * FROM A ORDER BY
        A.a ASC, A.b DESC, A.c NULLS FIRST, A.d ASC NULLS LAST

        """

        self.add('ORDER BY')
        fields = []
        for it in by:
            if isinstance(it, int):
                fields.append(SqlLiteral(str(it)))
            elif isinstance(it, str):
                fields.append(SqlLiteral(it))
            elif isinstance(it, SqlExpr):
                fields.append(it)
            else:
                fields.append(wrap(it))

        SqlConcatOper(fields, ',').__sql_stat__(self)

        return self

    def window_def(self, *windows: SqlWindowDef | SqlAlias[SqlWindowDef], **window_ks: SqlWindowDef) -> Self:
        """
        define windows.

        """
        if (defs := self._window_defs) is None:
            raise RuntimeError('not a select statement')

        for window in windows:
            if isinstance(window, SqlWindowDef):
                if window.name is None:
                    raise RuntimeError('?? AS ' + repr(window))
                defs[window.name] = window
            elif isinstance(window, SqlAlias) and isinstance(window.value, SqlWindowDef):
                defs[window.name] = window.value
            else:
                raise TypeError()

        defs.update(window_ks)
        return self

    @overload
    def join(self, table: type[S] | SqlSelectStat[S] | SqlAlias[S], *,
             by: Literal['left', 'right', 'inner', 'full outer'] = None) -> SqlJoinStat:
        pass

    @overload
    def join(self, table: type[S] | SqlSelectStat[S] | SqlAlias[S], *,
             by: Literal['cross']) -> SqlSelectStat[tuple]:
        pass

    def join(self, table: type[S] | SqlSelectStat[S] | SqlAlias[S], *,
             by: Literal['left', 'right', 'inner', 'full outer', 'cross'] = None):
        """
        ``JOIN`` https://www.sqlite.org/lang_select.html#strange_join_names

        >>> select_from(A.a, B.b).join(B).on(A.a == B.a) # doctest: SKIP
        SELECT A.a, B.b FROM A
        JOIN B ON A.a = B.a

        """
        if by is not None:
            self.add([by.upper(), 'JOIN'])

        if isinstance(table, type):
            self.add(table_name(table))
        elif isinstance(table, SqlStat):
            self.add(table)
        elif isinstance(table, SqlAlias) and isinstance(other := table.value, type):
            self.add(table_name(other))
            self.add(table.name)
        elif isinstance(table, SqlAlias) and isinstance(other := table.value, SqlSubQuery):
            self.add(other.stat)
            self.add('AS')
            self.add(table.name)
        else:
            raise TypeError()

        if by == 'cross':
            return self
        return SqlJoinStat(self)

    def group_by(self, *by) -> Self:
        """
        ``GROUP BY`` https://www.sqlite.org/lang_select.html#resultset
        """
        if len(by) == 0:
            raise RuntimeError()

        self.add('GROUP BY')
        for field in by:
            if isinstance(field, (int, float, bool)):
                self.add(str(field))
            elif isinstance(field, str):
                self.add(repr(field))
            elif isinstance(field, Field):
                self.add(f'{field.table_name}.{field.name}')
            elif isinstance(field, SqlField):
                self.add(f'{field.table_name}.{field.name}')
            elif isinstance(field, SqlAlias):
                self.add(field.name)
            elif isinstance(field, SqlExpr):
                field.__sql_stat__(self)
            else:
                raise TypeError('GROUP BY ' + repr(field))
            self.add(',')

        self._stat.pop()
        return self

    def having(self, *exprs: bool | SqlExpr) -> Self:
        """
        ``HAVING`` https://www.sqlite.org/lang_select.html#resultset
        """
        if len(exprs) == 0:
            return self

        from .func_stat import and_
        self.add('HAVING')
        and_(*exprs).__sql_stat__(self)

        return self

    def intersect(self, stat: SqlStat) -> Self:
        """
        ``INTERSECT`` https://www.sqlite.org/lang_select.html#compound_select_statements
        """
        self.add('INTERSECT')
        self.add(stat._stat, *stat._para)
        stat._connection = None
        return self

    def __and__(self, other: SqlStat) -> Self:
        """
        ``INTERSECT`` https://www.sqlite.org/lang_select.html#compound_select_statements
        """
        return self.intersect(other)

    def union(self, stat: SqlStat, all=False) -> Self:
        """
        ``UNION`` https://www.sqlite.org/lang_select.html#compound_select_statements
        """
        self.add('UNION')
        if all:
            self.add('ALL')
        self.add(stat._stat, *stat._para)
        stat._connection = None
        return self

    def __or__(self, other: SqlStat) -> Self:
        """
        ``UNION`` https://www.sqlite.org/lang_select.html#compound_select_statements
        """
        return self.union(other)

    def except_(self, stat: SqlStat) -> Self:
        """
        ``EXCEPT`` https://www.sqlite.org/lang_select.html#compound_select_statements
        """
        self.add('EXCEPT')
        self.add(stat._stat, *stat._para)
        stat._connection = None
        return self

    def __sub__(self, other: SqlStat) -> Self:
        """
        ``EXCEPT`` https://www.sqlite.org/lang_select.html#compound_select_statements
        """
        return self.except_(other)


class SqlJoinStat:
    def __init__(self, stat: SqlSelectStat[tuple]):
        self._stat = stat

    def using(self, *fields) -> SqlSelectStat[tuple]:
        if len(fields) == 0:
            raise RuntimeError()

        stat = self._stat
        stat.add(['USING', '('])

        for field in fields:
            if isinstance(field, str):
                stat.add(repr(field))
            elif isinstance(field, Field):
                stat.add(field.name)
            elif isinstance(field, SqlField):
                stat.add(field.name)
            else:
                raise TypeError('USING ' + repr(field))
            stat.add(',')

        stat._stat.pop()
        stat.add(')')

        return self._stat

    def on(self, *exprs: bool | SqlExpr) -> SqlSelectStat[tuple]:
        self._stat.add('ON')
        from .func_stat import and_
        and_(*exprs).__sql_stat__(self._stat)
        return self._stat


class SqlInsertStat(SqlStat[T], Generic[T]):
    def __init__(self, table: type[T], *, named: bool = False):
        super().__init__(table)
        self._values: dict[str, Any] = {
            name: f':{name}' if named else '?'
            for name in table_field_names(table)
        }
        self._named = named

    @overload
    def select_from(self, table: type[T], *, distinct: bool = False) -> SqlSelectStat[T]:
        pass

    @overload
    def select_from(self, *field, distinct: bool = False,
                    from_table: str | type | SqlAlias | SqlSelectStat = None) -> SqlSelectStat[tuple]:
        pass

    def select_from(self, *args, distinct: bool = False,
                    from_table: str | type | SqlAlias | SqlSelectStat = None) -> SqlSelectStat[T]:
        self._connection = None

        self.add('(')
        for i, field in enumerate(self._values):
            if i > 0:
                self.add(',')
            self.add(field)
        self.add(')')

        ret = select_from(*args, distinct=distinct, from_table=from_table)
        ret._stat = [*self._stat, *ret._stat]

        return ret

    def values(self, **kwargs: str | SqlExpr) -> Self:
        for field, expr in kwargs.items():
            if field not in self._values:
                raise RuntimeError(f'{table_name(self.table)}.{field} not found')
            self._values[field] = expr
        return self

    def build(self) -> str:
        self.add(['VALUES', '('])
        for field, expr in self._values.items():
            if isinstance(expr, str):
                self.add(expr)
            elif isinstance(expr, SqlStat):
                self.add(expr)
            elif isinstance(expr, SqlExpr):
                self._deparameter = True
                expr.__sql_stat__(self)
                self._deparameter = False
            else:
                raise TypeError()
            self.add(',')
        self._stat.pop()
        self.add(')')

        return super().build()

    def submit(self, parameter: list[T]) -> Cursor[T]:
        if (connection := self._connection) is None:
            raise RuntimeError('Do not in a connection context')

        from .table import _table_class
        try:
            table_cls = _table_class(type(parameter[0]))
        except AttributeError:
            pass
        else:
            if self._named:
                parameter = list(map(table_cls.table_dict, parameter))
            else:
                parameter = list(map(table_cls.table_seq, parameter))

        ret = Cursor(connection, connection.execute_batch(self, parameter, commit=True), self.table)
        self._connection = None
        return ret


@overload
def select_from(table: type[T], *, distinct: bool = False) -> SqlSelectStat[T]:
    """
    >>> select_from(Table) # SELECT * FROM Table
    """
    pass


@overload
def select_from(*field, distinct: bool = False,
                from_table: Union[str, type, SqlAlias, SqlSelectStat] = None) -> SqlSelectStat[tuple]:
    """
    >>> select_from('a', 'b') # SELECT a, b FROM Table
    """
    pass


def select_from(*args, distinct: bool = False,
                from_table: Union[str, type, SqlAlias, SqlSelectStat] = None) -> SqlSelectStat:
    """
    ``SELECT``: https://www.sqlite.org/lang_select.html

    Select all fields from a table

    >>> select_from(A).build() # doctest: SKIP
    SELECT * FROM A
    >>> select_from(A).fetchall() # doctest: SKIP
    [A(...), A(...), ...]

    Select subset of fields from A

    >>> select_from(A.a, A.b).build() # doctest: SKIP
    SELECT A.a, A.b FROM A
    >>> select_from(A.a, A.b).fetchall() # doctest: SKIP
    [('a', 1), ('b', 2), ...]

    With a literal value

    >>> select_from(A.a, 0).build() # doctest: SKIP
    SELECT A.a, 0 FROM A
    >>> select_from(A.a, 0).fetchall() # doctest: SKIP
    [('a', 0), ('b', 0), ...]

    With SQL functions

    >>> select_from(A.a, count()).build() # doctest: SKIP
    SELECT A.a, COUNT(*) FROM A

    Use table alias

    >>> a = alias(A, 'a') # doctest: SKIP
    >>> select_from(a.a).build() # doctest: SKIP
    SELECT a.a from A a

    join other tables

    >>> select_from(A.a, B.b).join(B).on(A.c == B.c).build() # doctest: SKIP
    SELECT A.a, B.b FROM A JOIN B ON A.c = B.c

    **features supporting**

    * `SELECT DISTINCT`
    * `FROM`
    * `WHERE`
    * `GROUP BY`
    * `HAVING`
    * `WINDOW`
    * compound-operator: `UNION [ALL]`, `INTERSECT` and `EXCEPT`
    * `ORDER BY`
    * `LIMIT [OFFSET]`

    **features not supporting**

    * `WITH [RECURSIVE]`
    * `SELECT ALL`
    * `VALUES`

    :param args:
    :param distinct:
    :param from_table:
    :return:
    """
    pre_stat = ['SELECT']
    if distinct:
        pre_stat.append('DISTINCT')

    if len(args) == 0:
        raise RuntimeError()
    elif len(args) == 1 and isinstance(table := args[0], type):
        self = SqlSelectStat(table, table_fields(table))
        self.add(pre_stat)
        self.add('*')
    else:
        headers = []
        self = SqlSelectStat(None, headers)
        self.add(pre_stat)

        table, fields = _select_from_fields(*args)
        if table is None:
            table = from_table
            if table is None:
                raise RuntimeError('need to provide from_table')

        for i, field in enumerate(fields):
            if i > 0:
                self.add(',')

            if isinstance(field, Field):
                self.add(f'{field.table_name}.{field.name}')
                headers.append(field)
            elif isinstance(field, SqlAlias) and isinstance(field.value, SqlField):
                name = field.name
                field = field.value.field
                self.add([f'{field.table_name}.{field.name}', 'AS', repr(name)])
                headers.append(name)
            elif isinstance(field, SqlAlias):
                field.value.__sql_stat__(self)
                self.add(['AS', repr(field.name)])
                headers.append(field.name)
            elif isinstance(field, SqlAliasField):
                field.__sql_stat__(self)
                headers.append(str(field))
            elif isinstance(field, SqlFuncOper):
                field.__sql_stat__(self)
                headers.append(f'{field.oper}()')
            elif isinstance(field, SqlLiteral):
                field.__sql_stat__(self)
                headers.append(None)
            elif isinstance(field, SqlExpr):
                field.__sql_stat__(self)
                headers.append(None)
            else:
                raise TypeError('SELECT ' + repr(field))

    self.add('FROM')
    if isinstance(table, str):
        self.add(table)
    elif isinstance(table, type):
        self.add(table_name(table))
    elif isinstance(table, SqlStat):
        self.add('(')
        self.add(table)
        self.add(')')
    elif isinstance(table, SqlAlias) and isinstance(table.value, type):
        self.add([table_name(table.value), table.name])
    elif isinstance(table, SqlAlias) and isinstance(table.value, SqlSubQuery):
        self.add('(')
        self.add(table.value.stat)
        self.add([')', 'AS', repr(table.name)])
    else:
        raise TypeError('FROM ' + repr(table))

    if len(self._window_defs):
        self.add('WINDOW')
        for i, (name, window) in enumerate(self._window_defs.items()):
            if i > 0:
                self.add(',')

            self.add([name, 'AS'])
            window.__sql_stat__(self)

    return self


def _select_from_fields(*args) -> tuple[type, list[Field]]:
    if len(args) == 0:
        raise RuntimeError('empty field')

    table = None
    fields = []
    for arg in args:
        if isinstance(arg, (int, float, bool, str)):
            fields.append(SqlLiteral(repr(arg)))
        elif isinstance(arg, SqlField):
            field = arg.field
            if table is None:
                table = field.table

            fields.append(field)
        elif isinstance(arg, SqlAlias) and isinstance(arg.value, SqlField):
            field = arg.value.field

            if table is None:
                table = field.table

            fields.append(arg)

        elif isinstance(arg, SqlAliasField) and isinstance(arg.table, type):
            if table is None:
                table = SqlAlias(arg.table, arg.name)

            fields.append(arg)

        elif isinstance(arg, SqlExpr):
            fields.append(arg)

        else:
            raise TypeError(repr(arg))

    return table, fields


def insert_into(table: type[T], *, policy: UPDATE_POLICY = None, named=False) -> SqlInsertStat[T]:
    """
    ``INSERT``: https://www.sqlite.org/lang_insert.html

    insert values

    >>> insert_into(A, policy='REPLACE').build() # doctest: SKIP
    INSERT OR REPLACE INTO A VALUES (?)
    >>> insert_into(A, policy='REPLACE').submit([A(1), A(2)]) # doctest: SKIP

    insert values with field overwrite

    >>> insert_into(A, policy='REPLACE').values(a='1').build() # doctest: SKIP
    INSERT OR REPLACE INTO A VALUES (1)

    insert values from a table

    >>> insert_into(A, policy='IGNORE').select_from(B).build() # doctest: SKIP
    INSERT OR IGNORE INTO A
    SELECT * FROM B

    **features supporting**

    * `INSERT [OR ...]`
    * `VALUES`
    * `SELECT`

    **features not supporting**

    * `WITH [RECURSIVE]`
    * `REPLACE`
    * `DEFAULT VALUES`
    * upsert clause
    * returning clause

    :param table:
    :param policy:
    :param named:
    :return:
    """
    self = SqlInsertStat(table, named=named)
    self.add('INSERT')
    if policy is not None:
        self.add(['OR', policy])
    self.add(['INTO', table_name(table)])
    return self


def update(table: type[T], *args: bool, **kwargs) -> SqlWhereStat[T]:
    """
    ``UPDATE``: https://www.sqlite.org/lang_update.html

    >>> update(A, A.a==1).where(A.b==2).build() # doctest: SKIP
    UPDATE A SET A.a = 1 WHERE A.b = 2

     **features supporting**

    * `UPDATE [OR ...]`
    * `SET COLUMN = EXPR`
    * `FROM`
    * `WHERE`

    **features not supporting**

    * `WITH [RECURSIVE]`
    * (qualified table name) `INDEXED BY`
    * (qualified table name) `NOT INDEXED`
    * `SET (COLUMNS) = EXPR`
    * returning clause

    :param table:
    :param args:
    :param kwargs:
    :return:
    """
    self = SqlWhereStat(table)
    self.add(['UPDATE', table_name(table), 'SET'])

    for arg in args:
        if isinstance(arg, SqlCompareOper) and arg.oper == '=' and isinstance(arg.left, SqlField):
            field = arg.left.field
            value = arg.right
            if isinstance(value, SqlPlaceHolder):
                self.add(f'{field.name} = ?', value.value)
            elif isinstance(value, SqlLiteral):
                self.add(f'{field.name} = {value.value}')
            else:
                raise TypeError(arg)
        else:
            raise TypeError(arg)
        self.add(',')

    for term, value in kwargs.items():
        table_field(table, term)
        self.add(f'{term} = ?', value)
        self.add(',')

    self._stat.pop()

    return self


def delete_from(table: type[T]) -> SqlWhereStat[T]:
    """
    ``DELETE``: https://www.sqlite.org/lang_delete.html

    >>> delete_from(A).where(A.b > 2).build()  # doctest: SKIP
    DELETE FROM A WHERE A.b > 2

    **features supporting**

    * `DELETE FROM`
    * `WHERE`
    * `ORDER BY`
    * `LIMIT [OFFSET]`

    **features not supporting**

    * `WITH [RECURSIVE]`
    * (qualified table name) `INDEXED BY`
    * (qualified table name) `NOT INDEXED`
    * returning clause

    :param table:
    :return:
    """
    self = SqlWhereStat(table)
    self.add(['DELETE', 'FROM', table_name(table)])
    return self


def create_table(table: type[T], *,
                 if_not_exists=True,
                 primary_policy: CONFLICT_POLICY = None,
                 unique_policy: CONFLICT_POLICY = None) -> SqlStat[T]:
    """
    ``CREATE``: https://www.sqlite.org/lang_createtable.html

    >>> @named_tuple_table_class # doctest: SKIP
    ... class A(NamedTuple):
    ...     a: int
    >>> create_table(A) # doctest: SKIP
    CREATE TABLE IF NOT EXISTS A (a INT NOT NULL)

    **features supporting**

    * `IF NOT EXISTS`
    * column constraint `NOT NULL`
    * column constraint `UNIQUE`
    * column constraint `CHECK`
    * column constraint `DEFAULT value`
    * table constraint `PRIMARY KEY`
    * table constraint `UNIQUE`
    * table constraint `CHECK`
    * table constraint `FOREIGN KEY`

    **features not supporting**

    * `CREATE TEMP|TEMPORARY`
    * `CREATE TEMP`
    * `AS SELECT`
    * column constraint `CONSTRAINT`
    * column constraint `PRIMARY KEY`
    * column constraint `NOT NULL ON CONFLICT`
    * column constraint `UNIQUE ON CONFLICT`
    * column constraint `DEFAULT (EXPR)`
    * column constraint `COLLATE`
    * column constraint `REFERENCES`
    * column constraint `[GENERATED ALWAYS] AS`
    * table constraint `CONSTRAINT`
    * `WITHOUT ROWID`
    * `STRICT`

    :param table:
    :param primary_policy:
    :param unique_policy:
    :return:
    """
    self = SqlStat(table)
    self.add(['CREATE', 'TABLE'])
    if if_not_exists:
        self.add('IF NOT EXISTS')
    self.add(table_name(table))
    self.add('(')

    for i, field in enumerate(table_fields(table)):
        _column_def(self, field)
        self.add(',')

    if len(primary_keys := table_primary_field_names(table)) > 0:
        self.add(['PRIMARY KEY', '(', ','.join(primary_keys), ')'])
        if primary_policy is not None:
            self.add(['ON CONFLICT', primary_policy.upper()])
        self.add(',')

    if len(unique_keys := table_unique_field_names(table)) > 0:
        self.add(['UNIQUE', '(', ','.join(unique_keys), ')'])
        if unique_policy is not None:
            self.add(['ON CONFLICT', unique_policy.upper()])
        self.add(',')

    for foreign_key in table_foreign_fields(table):
        _foreign_key(self, foreign_key)
        self.add(',')

    if (check := table_check_field(table, None)) is not None:
        self._deparameter = True
        self.add(['CHECK', '('])
        check.expression.__sql_stat__(self)
        self.add([')', ','])
        self._deparameter = False

    self._stat.pop()
    self.add(')')

    return self


def _column_def(self: SqlStat, field: Field):
    self.add(field.name)

    if field.f_type == int:
        self.add('INT')
    elif field.f_type == float:
        self.add('FLOAT')
    elif field.f_type == bool:
        self.add('BOOLEAN')
    elif field.f_type == bytes:
        self.add('BLOB')
    elif field.f_type == str:
        self.add('TEXT')
    elif field.f_type == datetime.date:
        self.add('DATETIME')
    elif field.f_type == datetime.datetime:
        self.add('DATETIME')
    else:
        raise RuntimeError(f'field type {field.f_type}')

    if field.not_null:
        if not field.has_default or field.f_value is not None:
            self.add('NOT NULL')

    if field.has_default:
        if field.f_value is None:
            self.add(f'DEFAULT NULL')
        else:
            self.add(f'DEFAULT {repr(field.f_value)}')

    if (check := table_check_field(field.table, field.name)) is not None:
        self._deparameter = True
        self.add(['CHECK', '('])
        check.expression.__sql_stat__(self)
        self.add(')')
        self._deparameter = False


def _foreign_key(self: SqlStat, foreign: ForeignConstraint):
    self.add(['FOREIGN KEY'])
    self.add('(')
    self.add(' , '.join(foreign.fields))
    self.add(')')
    self.add('REFERENCES')
    self.add(table_name(foreign.foreign_table))
    self.add('(')
    self.add(' , '.join(foreign.foreign_fields))
    self.add(')')
    if (policy := foreign.on_update) is not None:
        self.add(['ON UPDATE', policy])
    if (policy := foreign.on_delete) is not None:
        self.add(['ON DELETE', policy])
