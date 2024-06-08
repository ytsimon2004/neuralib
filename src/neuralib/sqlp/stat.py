from __future__ import annotations

import datetime
import sqlite3
import warnings
from collections.abc import Iterator
from typing import overload, TYPE_CHECKING, Any, TypeVar, Generic, Optional, Literal, Union, cast, Callable

import polars as pl
from typing_extensions import Self

from .expr import *
from .expr import sql_join_set
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
    'replace_into',
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
        self._stat: list[str] = []
        self._para: list[Any] = []
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

    def __str__(self) -> str:
        table = table_name(self.table) if self.table is not None else '...'
        return f'{type(Self).__name__}[{table}]'

    def __repr__(self) -> str:
        ret = []
        par = list(self._para)
        for value in self._stat:
            if value == '?':
                ret.append(f'?({repr(par.pop(0))})')
            else:
                ret.append(value)
        return ' '.join(ret)

    def submit(self) -> Cursor[T]:
        """
        build the SQL statement and execute.

        :return: a cursor
        :raise RuntimeError: current statement does not bind with a connection.
        """
        if (connection := self._connection) is None:
            raise RuntimeError('Do not in a connection context')

        ret = Cursor(connection, connection.execute(self), self.table)
        self._connection = None
        return ret

    def __del__(self):
        # check connection and auto_commit,
        # make SqlStat auto submit itself when nobody is referring to it,
        # so users do need to explict call submit() for every statements.
        if self._connection is not None and self._stat is not None and len(self._stat) > 0:
            try:
                self.submit()
            except BaseException as e:
                warnings.warn(repr(e))

    @overload
    def add(self, stat: SqlStat) -> Self:
        pass

    @overload
    def add(self, stat: Union[str, list[str]], *para) -> Self:
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

    def __init__(self, connection: Connection, cursor: sqlite3.Cursor, table: type[T] = None):
        self._connection = connection
        self._cursor = cursor
        self._table = table

        header = cursor.description
        self.headers: list[str] = [it[0] for it in header] if header is not None else []
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

    def fetchone(self) -> Optional[T]:
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


class SqlWhereStat:
    """statement with **WHERE** support."""

    def where(self, *expr: Union[bool, SqlCompareOper, SqlExpr, Any, None]) -> Self:
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
        zelf = cast(SqlStat, self)
        expr = [it for it in expr if it is not None]
        if len(expr):
            from .func_stat import and_
            zelf.add('WHERE')
            and_(*expr).__sql_stat__(zelf)
        return self


class SqlLimitStat:
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

        **NOTE**

        LIMIT on UPDATE/DELETE need compile flag ``SQLITE_ENABLE_UPDATE_DELETE_LIMIT``.

        >>> assert Connection().sqlite_compileoption_used('SQLITE_ENABLE_UPDATE_DELETE_LIMIT') # doctest: SKIP

        """
        zelf = cast(SqlStat, self)

        if len(args) == 1:
            n, *_ = args
            zelf.add(['LIMIT', str(n)])
        elif len(args) == 2:
            row_count, offset = args
            zelf.add(['LIMIT', str(row_count), 'OFFSET', str(offset)])
        else:
            raise TypeError()
        return self

    def order_by(self, *by: Union[int, str, SqlExpr, Any]) -> Self:
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

        **NOTE**

        ORDER BY on UPDATE/DELETE need compile flag ``SQLITE_ENABLE_UPDATE_DELETE_LIMIT``.

        >>> assert Connection().sqlite_compileoption_used('SQLITE_ENABLE_UPDATE_DELETE_LIMIT') # doctest: SKIP

        """
        zelf = cast(SqlStat, self)
        zelf.add('ORDER BY')
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

        SqlConcatOper(fields, ',').__sql_stat__(zelf)

        return self


class SqlSelectStat(SqlStat[T], SqlWhereStat, SqlLimitStat, Generic[T]):
    """**SELECT** statement."""

    def __init__(self, table: Optional[type[T]], this_table: Optional[type[T]]):
        super().__init__(table)
        self._this_table = this_table
        self._window_defs: dict[str, SqlWindowDef] = {}

    def __matmul__(self, other: str) -> SqlAlias[SqlSubQuery]:
        """wrap itself as a subquery with an alias name."""
        return SqlAlias(SqlSubQuery(self), other)

    def fetchall(self) -> list[T]:
        """submit and fetch all result."""
        return self.submit().fetchall()

    def fetchone(self) -> Optional[T]:
        """submit and fetch the first result."""
        return self.submit().fetchone()

    def __iter__(self) -> Iterator[T]:
        """submit and iterate the results"""
        return iter(self.submit())

    def fetch_polars(self) -> pl.DataFrame:
        """submit and fetch all as a polar dataframe."""
        return self.submit().fetch_polars()

    def window_def(self, *windows: Union[SqlWindowDef, SqlAlias[SqlWindowDef]], **window_ks: SqlWindowDef) -> Self:
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
            elif isinstance(window, SqlAlias) and isinstance(window._value, SqlWindowDef):
                defs[window.name] = window._value
            else:
                raise TypeError()

        defs.update(window_ks)
        return self

    @overload
    def join(self, table: Union[type[S], SqlSelectStat[S], SqlAlias[S]], *,
             by: Literal['left', 'right', 'inner', 'full outer'] = None) -> SqlJoinStat:
        pass

    @overload
    def join(self, table: Union[type[S], SqlSelectStat[S], SqlAlias[S]], *,
             by: Literal['cross']) -> SqlSelectStat[tuple]:
        pass

    def join(self, table: Union[type[S], SqlSelectStat[S], SqlAlias[S]], *,
             by: Literal['left', 'right', 'inner', 'full outer', 'cross'] = None):
        """
        ``JOIN`` https://www.sqlite.org/lang_select.html#strange_join_names

        >>> select_from(A.a, B.b).join(B).on(A.a == B.a) # doctest: SKIP
        SELECT A.a, B.b FROM A
        JOIN B ON A.a = B.a

        """
        if by is not None:
            self.add(by.upper())
        self.add('JOIN')

        if isinstance(table, type):
            that_table = table
            self.add(table_name(table))
        elif isinstance(table, SqlSelectStat):
            self.add(table)
            that_table = table._this_table
        elif isinstance(table, SqlAlias) and isinstance(that_table := table._value, type):
            self.add(table_name(that_table))
            self.add(table._name)
        elif isinstance(table, SqlAlias) and isinstance(table._value, SqlSubQuery) and isinstance(table._value.stat, SqlSelectStat):
            self.add(table._value.stat)
            self.add('AS')
            self.add(table._name)
            that_table = table._value.stat._this_table
        else:
            raise TypeError(f'JOIN {table}')

        self.table = None

        if by == 'cross':
            return self

        return SqlJoinStat(self, self._this_table, that_table)

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
                self.add(field._name)
            elif isinstance(field, SqlExpr):
                field.__sql_stat__(self)
            else:
                raise TypeError('GROUP BY ' + repr(field))
            self.add(',')

        self._stat.pop()
        return self

    def having(self, *exprs: Union[bool, SqlExpr]) -> Self:
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
    def __init__(self, stat: SqlSelectStat[tuple], this: type, that: Union[type, SqlStat]):
        self._stat = stat
        self._this = this
        self._that = that

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

        return stat

    def on(self, *exprs: Union[bool, SqlExpr]) -> SqlSelectStat[tuple]:
        if len(exprs) == 0:
            raise RuntimeError()

        self._stat.add('ON')
        from .func_stat import and_
        and_(*exprs).__sql_stat__(self._stat)
        return self._stat

    def using_foreign(self, other: Union[type, Callable] = None) -> SqlSelectStat[tuple]:
        this = self._this
        that = other or self._that

        if (constraint := table_foreign_field(this, that)) is None:
            if (constraint := table_foreign_field(that, this)) is None:
                raise RuntimeError(f'no foreign constraint between {this.__name__} and {that.__name__}')

        if constraint.fields != constraint.foreign_fields:
            raise RuntimeError('foreign constraint does not contain same field names')

        stat = self._stat

        stat.add(['USING', '('])

        for field in constraint.fields:
            stat.add(field)
            stat.add(',')

        stat._stat.pop()
        stat.add(')')

        return stat

    def on_foreign(self, other: Union[type, Callable] = None) -> SqlSelectStat[tuple]:
        this = self._this
        that = other or self._that

        if (constraint := table_foreign_field(this, that)) is None:
            if (constraint := table_foreign_field(that, this)) is None:
                raise RuntimeError(f'no foreign constraint between {this.__name__} and {that.__name__}')

        stat = self._stat

        stat.add(['ON', '('])
        for af, bf in zip(constraint.fields, constraint.foreign_fields):
            stat.add([f'{this.__name__}.{af} = {that.__name__}.{bf}'])
            stat.add('AND')

        stat._stat.pop()
        stat.add(')')

        return stat


class SqlReturnStat:
    def returning(self, *expr: Union[str, Any]) -> Self:
        zelf = cast(SqlStat, self)
        zelf.add('RETURNING')
        if len(expr) == 0:
            zelf.add('*')
        else:
            fields = []
            for exp in expr:
                if isinstance(exp, str):
                    zelf.add(exp)
                    fields.append(exp)
                elif isinstance(exp, SqlField):
                    zelf.add(exp.name)
                    fields.append(exp.name)
                elif isinstance(exp, SqlAlias) and isinstance(field := exp._value, SqlField):
                    zelf.add([field.name, 'AS', exp._name])
                    fields.append(exp._name)
                elif isinstance(exp, SqlAlias) and isinstance(_expr := exp._value, SqlExpr):
                    zelf._deparameter = True
                    _expr.__sql_stat__(zelf)
                    zelf._deparameter = False

                    zelf.add(['AS', exp._name])
                    fields.append(exp._name)
                else:
                    raise TypeError(f'RETURNING ({exp})')

                zelf.add(',')
            zelf._stat.pop()

            if isinstance(self, SqlInsertStat):
                self.table = None
                self._fields = fields

        return self


class SqlInsertStat(SqlStat[T], SqlReturnStat, Generic[T]):
    def __init__(self, table: type[T], fields: list[str] = None, *, named: bool = False):
        super().__init__(table)
        self._fields = fields

        # when
        #   None: `VALUES` set
        #   'DEFAULT': `DEFAULT VALUES` unset
        #   {field->value}: `DEFAULT VALUES` set
        self._values: Union[Literal['DEFAULT'], dict[str, Any], None] = {
            name: SqlLiteral(f':{name}') if named else SqlLiteral('?')
            for name in table_field_names(table)
        }
        self._named = named
        self._returning = False

    @overload
    def select_from(self, table: type[T], *, distinct: bool = False) -> SqlSelectStat[T]:
        pass

    @overload
    def select_from(self, *field, distinct: bool = False,
                    from_table: Union[str, type, SqlAlias, SqlSelectStat] = None) -> SqlSelectStat[tuple]:
        pass

    def select_from(self, *args, distinct: bool = False,
                    from_table: Union[str, type, SqlAlias, SqlSelectStat] = None) -> SqlSelectStat[T]:
        if isinstance(self._values, str):
            raise RuntimeError()

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

    def values(self, *args, **kwargs: Union[str, SqlExpr]) -> Self:
        # TODO support direct set values
        if self._values is None or isinstance(self._values, str):
            raise RuntimeError()

        for expr in args:
            if isinstance(expr, SqlCompareOper) and expr.oper == '=' and isinstance(expr.left, SqlField):
                self._values[expr.left.field.name] = expr.right
            else:
                raise TypeError(expr)

        for field, expr in kwargs.items():
            if field not in self._values:
                raise RuntimeError(f'{table_name(self.table)}.{field} not found')
            self._values[field] = expr
        return self

    def defaults(self) -> Self:
        """insert default values"""
        self._values = 'DEFAULT'
        return self

    def _set_values(self):
        if isinstance(self._values, str):
            self.add(['DEFAULT', 'VALUES'])
        elif isinstance(self._values, dict):
            if self._fields is None:
                fields = list(self._values.keys())
            else:
                fields = self._fields

            self.add(['VALUES', '('])

            for field in fields:
                value = self._values[field]
                if isinstance(value, (int, float, bool, str)):
                    self.add(repr(value))
                elif isinstance(value, SqlLiteral):
                    self.add(value.value)
                elif isinstance(value, SqlPlaceHolder):
                    self.add(repr(value.value))
                elif isinstance(value, SqlStat):
                    self.add(value)
                elif isinstance(value, SqlExpr):
                    self._deparameter = True
                    value.__sql_stat__(self)
                    self._deparameter = False
                else:
                    raise TypeError(repr(value))
                self.add(',')
            self._stat.pop()
            self.add(')')

        self._values = None

    def on_conflict(self, *conflict, where: Union[bool, SqlCompareOper] = None) -> SqlUpsertStat[T]:
        self._set_values()
        return SqlUpsertStat(self, *conflict, where=where)

    def returning(self, *expr: Union[str, SqlExpr]) -> SqlInsertStat[tuple]:
        self._set_values()
        super().returning(*expr)
        self._returning = True
        return self

    def build(self) -> str:
        self._set_values()
        return super().build()

    def submit(self, parameter: list[T] = ()) -> Cursor[T]:
        if (connection := self._connection) is None:
            raise RuntimeError('Do not in a connection context')

        from .table import _table_class
        try:
            table_cls = _table_class(self.table)
        except AttributeError:
            pass
        else:
            if self._named:
                def mapper(p):
                    if isinstance(p, self.table):
                        return table_cls.table_dict(p)
                    return dict(p)
            else:
                def mapper(p):
                    if isinstance(p, self.table):
                        return table_cls.table_seq(p)
                    return tuple(p)

            parameter = list(map(mapper, parameter))

        if len(parameter):
            if self._returning:
                if len(parameter) > 1:
                    raise RuntimeError('only support return one data')
                cur = connection.execute(self, parameter[0])
            else:
                cur = connection.execute_batch(self, parameter)
        else:
            cur = connection.execute(self, parameter)

        ret = Cursor(connection, cur, self.table)
        self._connection = None
        return ret


class SqlUpsertStat(Generic[T]):
    """

    https://www.sqlite.org/syntax/upsert-clause.html

    """

    def __init__(self, stat: SqlInsertStat[T], *conflict, where: Union[bool, SqlCompareOper] = None):
        self._stat = stat
        self._conflict = conflict
        self._stat.add(['ON', 'CONFLICT'])
        self._do_where = False

        if len(conflict):
            table, fields = _select_from_fields(*conflict)
            if table is None:
                table = stat.table
            elif table != stat.table:
                raise RuntimeError()

            self._stat.add('(')
            for i, field in enumerate(fields):
                if i > 0:
                    self._stat.add(',')

                if isinstance(field, SqlField):
                    if field.table != table:
                        raise RuntimeError(f'field {field.table_name}.{field.name} not belong to {table.__name__}')
                    self._stat.add(f'{field.name}')
                elif isinstance(field, SqlLiteral) and isinstance(field.value, str):
                    self._stat.add(field.value)
                else:
                    raise RuntimeError(f'ON CONFLICT ({field}:{type(field).__name__})')

            if where is not None:
                self._stat.add('WHERE')
                self._stat._deparameter = True
                where.__sql_stat__(self._stat)
                self._stat._deparameter = False
            self._stat.add(')')

    def do_nothing(self) -> SqlInsertStat[T]:
        self._stat.add(['DO', 'NOTHING'])
        return self._stat

    def do_update(self, *args: Union[bool, SqlCompareOper], where: Union[bool, SqlCompareOper] = None) -> SqlInsertStat[T]:
        self._stat.add(['DO', 'UPDATE', 'SET'])

        self._stat._deparameter = True  # TODO
        sql_join_set(self._stat, ',', args)
        self._stat._deparameter = False

        if where is not None:
            self._stat.add('WHERE')
            self._stat._deparameter = True
            where.__sql_stat__(self._stat)
            self._stat._deparameter = False
        return self._stat


class SqlUpdateStat(SqlStat[T], SqlWhereStat, SqlLimitStat, SqlReturnStat, Generic[T]):
    def from_(self, query: Union[SqlStat, SqlAlias[SqlSubQuery]]) -> Self:
        self.add('FROM')
        if isinstance(query, SqlStat):
            self.add(query)
        elif isinstance(query, SqlSubQuery):
            self.add(query.stat)
        elif isinstance(query, SqlAlias) and isinstance(query._value, SqlSubQuery):
            self.add(query._value.stat)
            self.add(['AS', query._name])
        else:
            raise TypeError(repr(query))
        return self


class SqlDeleteStat(SqlStat[T], SqlWhereStat, SqlLimitStat, SqlReturnStat, Generic[T]):
    pass


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
        self = SqlSelectStat(table, table)
        self.add(pre_stat)
        self.add('*')
    else:

        table, fields = _select_from_fields(*args)
        if table is None:
            if from_table is None:
                raise RuntimeError('need to provide from_table')
            table = from_table

        if isinstance(table, type):
            self = SqlSelectStat(None, table)
        elif isinstance(table, SqlAlias) and isinstance(table._value, type):
            self = SqlSelectStat(None, table._value)
        else:
            self = SqlSelectStat(None, None)
        self.add(pre_stat)

        for i, field in enumerate(fields):
            if i > 0:
                self.add(',')

            if isinstance(field, SqlField):
                self.add(f'{field.table_name}.{field.name}')
            elif isinstance(field, SqlAlias) and isinstance(field._value, SqlField):
                name = field._name
                field = field._value
                self.add([f'{field.table_name}.{field.name}', 'AS', repr(name)])
            elif isinstance(field, SqlAlias) and isinstance(field._value, SqlExpr):
                field._value.__sql_stat__(self)
                self.add(['AS', repr(field._name)])
            elif isinstance(field, SqlAliasField):
                field.__sql_stat__(self)
            elif isinstance(field, SqlFuncOper):
                field.__sql_stat__(self)
            elif isinstance(field, SqlLiteral):
                field.__sql_stat__(self)
            elif isinstance(field, SqlExpr):
                field.__sql_stat__(self)
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
    elif isinstance(table, SqlAlias) and isinstance(table._value, type):
        self.add([table_name(table._value), table._name])
    elif isinstance(table, SqlAlias) and isinstance(table._value, SqlSubQuery):
        self.add('(')
        self.add(table._value.stat)
        self.add([')', 'AS', repr(table._name)])
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


def _select_from_fields(*args) -> tuple[Union[type, SqlAlias, None], list[SqlExpr]]:
    if len(args) == 0:
        raise RuntimeError('empty field')

    table = None
    fields = []
    for arg in args:
        if isinstance(arg, (int, float, bool, str)):
            fields.append(SqlLiteral(repr(arg)))

        elif isinstance(arg, SqlField):
            if table is None:
                table = arg.table

            fields.append(arg)
        elif isinstance(arg, SqlAlias) and isinstance(arg._value, SqlField):
            if table is None:
                table = arg._value.table

            fields.append(arg)

        elif isinstance(arg, SqlAliasField) and isinstance(arg.table, type):
            if table is None:
                table = SqlAlias(arg.table, arg.name)

            fields.append(arg)

        elif isinstance(arg, SqlExpr):
            expr_table = _expr_use_table(arg)
            if table is None:
                table = expr_table

            fields.append(arg)

        else:
            raise TypeError(repr(arg))

    return table, fields


def _expr_use_table(expr: SqlExpr) -> Optional[type]:
    if isinstance(expr, SqlField):
        return expr.field.table
    elif isinstance(expr, SqlAlias):
        if isinstance(expr._value, type):
            return expr._value
        elif isinstance(expr._value, SqlExpr):
            return _expr_use_table(expr._value)
    elif isinstance(expr, SqlExistsOper):
        return expr.stat.table
    elif isinstance(expr, SqlCompareOper):
        return _expr_use_table(expr.left) or _expr_use_table(expr.right)
    elif isinstance(expr, SqlUnaryOper):
        return _expr_use_table(expr.right)
    elif isinstance(expr, SqlCastOper):
        return _expr_use_table(expr.right)
    elif isinstance(expr, SqlBinaryOper):
        return _expr_use_table(expr.left) or _expr_use_table(expr.right)
    elif isinstance(expr, SqlVarArgOper):
        for arg in expr.args:
            if (ret := _expr_use_table(arg)) is not None:
                return ret
    elif isinstance(expr, SqlConcatOper):
        for arg in expr.args:
            if (ret := _expr_use_table(arg)) is not None:
                return ret
    elif isinstance(expr, SqlFuncOper):
        for arg in expr.args:
            if (ret := _expr_use_table(arg)) is not None:
                return ret

    return None


@overload
def insert_into(table: type[T], *, policy: UPDATE_POLICY = None, named=False) -> SqlInsertStat[T]:
    pass


@overload
def insert_into(*field, policy: UPDATE_POLICY = None, named=False) -> SqlInsertStat[T]:
    pass


def insert_into(*args, policy: UPDATE_POLICY = None, named=False) -> SqlInsertStat[T]:
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
    * `DEFAULT VALUES`
    * `SELECT`
    * upsert clause
    * returning clause

    **features not supporting**

    * `WITH [RECURSIVE]`

    :param table:
    :param policy:
    :param named:
    :return:
    """
    if len(args) == 0:
        raise RuntimeError()
    elif len(args) == 1 and isinstance(args[0], type):
        self = SqlInsertStat((table := args[0]), named=named)
    else:
        table, fields = _select_from_fields(*args)
        if table is None:
            raise RuntimeError()

        for i, field in enumerate(list(fields)):
            if isinstance(field, SqlField):
                if field.table != table:
                    raise RuntimeError(f'field {field.table_name}.{field.name} not belong to {table.__name__}')
                fields[i] = field.name
            else:
                raise TypeError()

        self = SqlInsertStat(table, fields, named=named)

    self.add('INSERT')
    if policy is not None:
        self.add(['OR', policy])
    self.add(['INTO', table_name(table)])
    if self._fields is not None:
        self.add('(')
        for i, field in enumerate(self._fields):
            if i > 0:
                self.add(',')
            self.add(field)
        self.add(')')
    return self


@overload
def replace_into(table: type[T], *, named=False) -> SqlInsertStat[T]:
    pass


@overload
def replace_into(*field: Any, named=False) -> SqlInsertStat[T]:
    pass


def replace_into(*args, named=False) -> SqlInsertStat[T]:
    return insert_into(*args, policy='REPLACE', named=named)


def update(table: type[T], *args: Union[bool, SqlCompareOper], **kwargs) -> SqlUpdateStat[T]:
    """
    ``UPDATE``: https://www.sqlite.org/lang_update.html

    >>> update(A, A.a==1).where(A.b==2).build() # doctest: SKIP
    UPDATE A SET A.a = 1 WHERE A.b = 2

     **features supporting**

    * `UPDATE [OR ...]`
    * `SET COLUMN = EXPR`
    * `FROM`
    * `WHERE`
    * `ON CONFLICT (COLUMNS) SET (COLUMNS) = EXPR`
    * returning clause

    **features not supporting**

    * `WITH [RECURSIVE]`
    * (qualified table name) `INDEXED BY`
    * (qualified table name) `NOT INDEXED`

    :param table:
    :param args:
    :param kwargs:
    :return:
    """
    self = SqlUpdateStat(table)
    self.add(['UPDATE', table_name(table), 'SET'])

    if len(args):
        sql_join_set(self, ',', args)
        self.add(',')

    for term, value in kwargs.items():
        table_field(table, term)
        self.add([term, '=', '?', ','], value)
    self._stat.pop()

    return self


def delete_from(table: type[T]) -> SqlDeleteStat[T]:
    """
    ``DELETE``: https://www.sqlite.org/lang_delete.html

    >>> delete_from(A).where(A.b > 2).build()  # doctest: SKIP
    DELETE FROM A WHERE A.b > 2

    **features supporting**

    * `DELETE FROM`
    * `WHERE`
    * `ORDER BY`
    * `LIMIT [OFFSET]`
    * returning clause

    **features not supporting**

    * `WITH [RECURSIVE]`
    * (qualified table name) `INDEXED BY`
    * (qualified table name) `NOT INDEXED`

    :param table:
    :return:
    """
    self = SqlDeleteStat(table)
    self.add(['DELETE', 'FROM', table_name(table)])
    return self


# TODO https://www.sqlitetutorial.net/sqlite-cte/

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
    * column constraint `PRIMARY KEY`
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
    * column constraint `NOT NULL ON CONFLICT`
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

    n_primary_key = len(primary_keys := table_primary_fields(table))

    for i, field in enumerate(table_fields(table)):
        if n_primary_key == 1 and field.is_primary:
            _column_def(self, field, field.get_primary())
        else:
            _column_def(self, field)
        self.add(',')

    if n_primary_key > 1:
        self.add(['PRIMARY KEY', '(', ' , '.join([it.name for it in primary_keys]), ')'])
        if primary_policy is not None:
            self.add(['ON CONFLICT', primary_policy.upper()])
        self.add(',')

    for unique in table_unique_fields(table):
        if len(unique.fields) > 1:
            self.add(['UNIQUE', '(', ' , '.join(unique.fields), ')'])
            if unique.conflict is not None:
                self.add(['ON CONFLICT', unique.conflict.upper()])
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


def _column_def(self: SqlStat, field: Field, primary: PRIMARY = None):
    self.add(field.name)

    if field.sql_type == Any:
        pass
    elif field.sql_type == int:
        self.add('INTEGER')
    elif field.sql_type == float:
        self.add('FLOAT')
    elif field.sql_type == bool:
        self.add('BOOLEAN')
    elif field.sql_type == bytes:
        self.add('BLOB')
    elif field.sql_type == str:
        self.add('TEXT')
    elif field.sql_type == datetime.time:
        self.add('DATETIME')
    elif field.sql_type == datetime.date:
        self.add('DATETIME')
    elif field.sql_type == datetime.datetime:
        self.add('DATETIME')
    else:
        raise RuntimeError(f'field type {field.sql_type}')

    if field.not_null:
        if not field.has_default or field.f_value is not None:
            self.add('NOT NULL')

    if primary is not None:
        self.add(['PRIMARY', 'KEY'])
        if primary.order is not None:
            self.add(primary.order.upper())
        if primary.conflict is not None:
            self.add(['ON CONFLICT', primary.conflict.upper()])
        if primary.auto_increment:
            self.add('AUTOINCREMENT')
    elif (unique := field.get_unique()) is not None:
        self.add('UNIQUE')
        if unique.conflict is not None:
            self.add(['ON CONFLICT', unique.conflict.upper()])

    if field.has_default:
        if field.f_value is None:
            self.add(f'DEFAULT NULL')
        elif field.f_value == CURRENT_DATE:
            self.add(f'DEFAULT CURRENT_DATE')
        elif field.f_value == CURRENT_TIME:
            self.add(f'DEFAULT CURRENT_TIME')
        elif field.f_value == CURRENT_DATETIME:
            self.add(f'DEFAULT CURRENT_DATETIME')
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
