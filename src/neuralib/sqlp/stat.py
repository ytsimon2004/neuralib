from __future__ import annotations

import sqlite3
import warnings
from collections.abc import Iterator
from typing import overload, TYPE_CHECKING, Any, TypeVar, Generic, Optional, Literal, Union, cast, Callable

import polars as pl
from typing_extensions import Self

from .expr import *
from .expr import sql_join_set
from .table import *

if TYPE_CHECKING:
    from .connection import Connection

__all__ = [
    'SqlStat',
    'SqlSelectStat',
    'SqlInsertStat',
    'SqlUpdateStat',
    'SqlDeleteStat',
    'Cursor',
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
            from .table import table_class
            try:
                self._table_cls = table_class(table)
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

    def __init__(self, table: Optional[type[T]]):
        super().__init__(table)
        self._involved: list[Union[type, SqlAlias[type]]] = []
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
            self.add(table_name(table))
            that = [table]
        elif isinstance(table, SqlSelectStat):
            self.add(table)
            that = []
        elif isinstance(table, SqlAlias) and isinstance(that_table := table._value, type):
            self.add(table_name(that_table))
            self.add(table._name)
            that = [table]
        elif isinstance(table, SqlAlias) and isinstance(table._value, SqlSubQuery) and isinstance(table._value.stat, SqlSelectStat):
            self.add(table._value.stat)
            self.add('AS')
            self.add(table._name)
            that = []
        else:
            raise TypeError(f'JOIN {table}')

        self.table = None

        if by == 'cross':
            self._involved.extend(that)
            return self

        return SqlJoinStat(self, that)

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
    def __init__(self, stat: SqlSelectStat[tuple], join: list[Union[type, SqlAlias[type]]]):
        self._stat = stat
        self._this = stat._involved
        self._that = join

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

        stat._involved.extend(self._that)
        return stat

    def on(self, *exprs: Union[bool, SqlExpr]) -> SqlSelectStat[tuple]:
        if len(exprs) == 0:
            raise RuntimeError()

        self._stat.add('ON')
        from .func_stat import and_
        and_(*exprs).__sql_stat__(self._stat)

        self._stat._involved.extend(self._that)
        return self._stat

    def by(self, constraint: Union[Callable, ForeignConstraint]) -> SqlSelectStat[tuple]:
        if not isinstance(constraint, ForeignConstraint):
            if (constraint := table_foreign_field(constraint)) is None:
                raise RuntimeError('not a foreign constraint')

        stat = self._stat

        if constraint.fields == constraint.foreign_fields:
            stat.add(['USING', '('])

            for field in constraint.fields:
                stat.add(field)
                stat.add(',')

            stat._stat.pop()
            stat.add(')')

        else:
            this = that = None
            for _this in self._this:
                if isinstance(_this, type):
                    if _this == constraint.table:
                        this = _this.__name__
                        break
                    elif _this == constraint.foreign_table:
                        that = _this.__name__
                        break

                if isinstance(_this, SqlAlias) and (_type := _this._value, type):
                    if _type == constraint.table:
                        this = _this._name
                        break
                    elif _type == constraint.foreign_table:
                        that = _this._name
                        break

                raise TypeError('internal error')
            else:
                raise RuntimeError('improper foreign constraint')

            for _that in self._that:
                if isinstance(_that, type):
                    if this is None and _that == constraint.table:
                        this = _this.__name__
                        break
                    if that is None and _that == constraint.foreign_table:
                        that = _this.__name__
                        break
                if isinstance(_that, SqlAlias) and isinstance(_type := _that._value, type):
                    if this is None and _type == constraint.table:
                        this = _that._name
                        break
                    if that is None and _type == constraint.foreign_table:
                        that = _that._name
                        break
                raise TypeError('internal error')
            else:
                raise RuntimeError('improper foreign constraint')

            assert this is not None and that is not None

            stat.add(['ON', '('])
            for af, bf in zip(constraint.fields, constraint.foreign_fields):
                stat.add([f'{this}.{af} = {that}.{bf}'])
                stat.add('AND')

            stat._stat.pop()
            stat.add(')')

        stat._involved.extend(self._that)
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

        from .stat_start import select_from
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

        from .table import table_class
        try:
            table_cls = table_class(self.table)
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
            from .stat_start import select_from_fields
            table, fields = select_from_fields(*conflict)
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
