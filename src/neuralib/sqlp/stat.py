from __future__ import annotations

import functools
import sqlite3
import warnings
from collections.abc import Iterator
from typing import overload, TYPE_CHECKING, Any, TypeVar, Generic, Optional, Literal, Union, cast, Callable

import polars as pl
from typing_extensions import Self

from .expr import *
from .expr import SqlStatBuilder, SqlRemovePlaceHolder
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


def catch_error(f=None, *, attr: str = None):
    def _catch_error_decorator(f):
        @functools.wraps(f)
        def _catch_error(self, *args, **kwargs):
            if attr is None:
                stat: SqlStat = self
            else:
                stat = getattr(self, attr)

            with stat:
                return f(self, *args, **kwargs)

        return _catch_error

    if f is None:
        return _catch_error_decorator
    else:
        return _catch_error_decorator(f)


class SqlStat(Generic[T]):
    """Abstract SQL statement."""

    def __init__(self, table: Optional[type[T]]):
        self.table = table
        self._stat: list[Any] = []

        from .connection import get_connection_context
        self._connection: Optional[Connection] = get_connection_context()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.drop()

    def build(self) -> tuple[str, list]:
        """build a SQL statement."""
        builder = SqlStatBuilder()
        builder.add(self)
        self._stat = None
        return builder.build(), builder.para

    def __str__(self) -> str:
        table = table_name(self.table) if self.table is not None else '...'
        return f'{type(Self).__name__}[{table}]'

    def __repr__(self) -> str:
        builder = SqlStatBuilder()
        builder._deparameter = 'repr'
        builder.add(self)
        return builder.build()

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

    def drop(self):
        self._connection = None
        self._stat = None

    def __del__(self):
        # check connection and auto_commit,
        # make SqlStat auto submit itself when nobody is referring to it,
        # so users do need to explict call submit() for every statements.
        # FIXME error raised during statement constructing should be aware
        if self._connection is not None and self._stat is not None and len(self._stat) > 0:
            try:
                self.submit()
            except BaseException as e:
                warnings.warn(repr(e))

    def add(self, stat) -> Self:
        """
        Add SQL token.
        """
        if self._stat is None:
            raise RuntimeError("Statement is closed.")

        self._stat.append(stat)
        if isinstance(stat, SqlStat):
            stat._connection = None

        return self


class Cursor(Generic[T]):
    """
    A SQL cursor wrapper.

    It will try to cast to T from the tuple returns.

    """

    def __init__(self, connection: Connection, cursor: sqlite3.Cursor, table: type[T] = None):
        self._connection = connection
        self._cursor = cursor

        if table is not None:
            from .table import table_class
            try:
                table_cls = table_class(table)
            except AttributeError:
                pass
            else:
                cursor.row_factory = lambda _, row: table_cls.table_new(*row)

    @property
    def headers(self) -> list[str]:
        header = self._cursor.description
        return [it[0] for it in header] if header is not None else []

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

        return ret

    def __iter__(self) -> Iterator[T]:
        """iterate the results."""
        yield from iter(self._cursor)

    def fetch_polars(self) -> pl.DataFrame:
        return pl.DataFrame(list(self._cursor), schema=self.headers)


class SqlWhereStat:
    """statement with **WHERE** support."""

    @catch_error
    def where(self, *expr: Union[bool, SqlCompareOper, SqlExpr, None]) -> Self:
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
            zelf.add(and_(*expr))
        return self


class SqlLimitStat:
    @overload
    def limit(self, n: int) -> Self:
        pass

    @overload
    def limit(self, row_count: int, offset: int) -> Self:
        pass

    @catch_error
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

    @catch_error
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
            elif isinstance(it, SqlAlias):
                fields.append(SqlLiteral(it._name))
            elif isinstance(it, SqlExpr):
                fields.append(it)
            else:
                fields.append(wrap(it))

        zelf.add(SqlConcatOper(fields, ','))

        return self


class SqlSelectStat(SqlStat[T], SqlWhereStat, SqlLimitStat, Generic[T]):
    """**SELECT** statement."""

    def __init__(self, table: Optional[type[T]]):
        super().__init__(table)
        self._involved: list[Union[type, SqlAlias[type]]] = []

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

    @catch_error
    def windows(self, *windows: Union[SqlWindowDef, SqlAlias[SqlWindowDef]], **window_ks: SqlWindowDef) -> Self:
        """
        define windows.

        """
        if len(windows) == 0 and len(window_ks) == 0:
            return self

        self.add('WINDOW')

        for window in windows:
            if isinstance(window, SqlWindowDef):
                if window.name is None:
                    raise RuntimeError('?? AS ' + repr(window))
                self.add([window.name, 'AS'])
                self.add(window)
            elif isinstance(window, SqlAlias) and isinstance(window._value, SqlWindowDef):
                self.add([window.name, 'AS'])
                self.add(window)
            else:
                raise TypeError()

            self.add(',')

        for name, window in window_ks.items():
            self.add([name, 'AS'])
            self.add(window)
            self.add(',')
        self._stat.pop()

        return self

    BY = Literal['left', 'right', 'inner', 'full outer', 'cross']

    @overload
    def join(self, constraint: Union[Callable, ForeignConstraint], *, by: BY = None) -> SqlSelectStat[tuple]:
        pass

    @overload
    def join(self, table: Union[type[S], SqlAlias[S]],
             constraint: Union[Callable, ForeignConstraint], *,
             by: BY = None) -> SqlSelectStat[tuple]:
        pass

    @overload
    def join(self, table: Union[type[S], SqlSelectStat[S], SqlAlias[S], SqlCteExpr],
             *field: Union[bool, Any],
             by: BY = None) -> SqlSelectStat[tuple]:
        pass

    @overload
    def join(self, *field: bool | Any, by: BY = None) -> SqlSelectStat[tuple]:
        pass

    @catch_error
    def join(self, *args, by: BY = None) -> SqlSelectStat[tuple]:
        """
        ``JOIN`` https://www.sqlite.org/lang_select.html#strange_join_names

        >>> select_from(A.a, B.b).join(A.a == B.a) # doctest: SKIP
        SELECT A.a, B.b FROM A
        JOIN B ON A.a = B.a

        """
        if by is not None:
            self.add(by.upper())
        self.add('JOIN')
        self.table = None

        if len(args) == 2 and isinstance(table := args[0], type) and (
                callable(constraint := args[1]) or isinstance(constraint, ForeignConstraint)):
            if not isinstance(constraint, ForeignConstraint):
                if (constraint := table_foreign_field(constraint)) is None:
                    raise RuntimeError('not a foreign constraint')
            self.__join_foreign(constraint, table)

        elif len(args) > 0 and isinstance(table := args[0], type):
            self.__join(table, *args[1:])

        elif len(args) > 0 and isinstance(expr := args[0], SqlCteExpr):
            self._stat.insert(0, expr)
            self.__join(expr, *args[1:])

        elif len(args) > 0 and isinstance(stat := args[0], SqlSelectStat):
            self.add(stat)
            self.__join(None, *args[1:])

        elif len(args) == 2 and isinstance(table := args[0], SqlAlias) and (
                callable(constraint := args[1]) or isinstance(constraint, ForeignConstraint)):
            if not isinstance(constraint, ForeignConstraint):
                if (constraint := table_foreign_field(constraint)) is None:
                    raise RuntimeError('not a foreign constraint')
            self.__join_foreign(constraint, table)

        elif len(args) > 0 and isinstance(table := args[0], SqlAlias) and isinstance(table._value, type) and isinstance(
                table._name, str):
            self.__join(table, *args[1:])

        elif len(args) > 0 and isinstance(table := args[0], SqlAlias) and isinstance(expr := table._value,
                                                                                     SqlCteExpr) and isinstance(table._name, str):
            self._stat.insert(0, expr)
            self.__join(table, *args[1:])

        elif len(args) > 0 and isinstance(table := args[0], SqlAlias) \
                and isinstance(subq := table._value, SqlSubQuery) \
                and isinstance(name := table._name, str) \
                and isinstance(stat := subq.stat, SqlSelectStat):
            self.add(stat)
            self.add('AS')
            self.add(name)
            self.__join(None, *args[1:])

        elif len(args) == 1 and isinstance(constraint := args[0], ForeignConstraint):
            self.__join_foreign(constraint, None)

        elif len(args) == 1 and callable(constraint := args[0]):
            if (constraint := table_foreign_field(constraint)) is None:
                raise RuntimeError('not a foreign constraint')
            self.__join_foreign(constraint, None)

        else:
            table = None
            for field in args:
                if table is None and isinstance(field, SqlExpr):
                    table = use_table(field, self._involved)
            if table is None:
                raise RuntimeError('no join table')
            self.__join(table, *args)

        return self

    def __join(self, right: Union[type, SqlAlias, SqlCteExpr, None], *fields):
        if right is None:
            pass
        elif isinstance(right, type):
            self.add(table_name(right))
        elif isinstance(right, SqlCteExpr):
            self.add(right._name)
        elif isinstance(right, SqlAlias) and isinstance(table := right._value, type):
            self.add(table_name(table))
            self.add(right._name)
        elif isinstance(right, SqlAlias) and isinstance(expr := right._value, SqlCteExpr):
            self.add(expr._select)
            self.add(right._name)
        else:
            raise TypeError(f'JOIN {right}')

        if len(fields):
            if any([isinstance(it, SqlCompareOper) for it in fields]):
                self.__join_on(*fields)
            else:
                self.__join_use(*fields)

        self._involved.append(right)

    def __join_use(self, *fields):
        self.add(['USING', '('])

        for field in fields:
            if isinstance(field, str):
                self.add(repr(field))
            if isinstance(field, Field):
                self.add(field.name)
            if isinstance(field, SqlField):
                self.add(field.name)
            else:
                raise TypeError('USING ' + repr(field))
            self.add(',')

        self._stat.pop()
        self.add(')')

    def __join_on(self, *exprs: bool | SqlExpr):
        self.add('ON')
        from .func_stat import and_
        self.add(and_(*exprs))

    def __join_foreign(self, constraint: ForeignConstraint, right: type | SqlAlias | None):
        this = that = None

        for _this in self._involved:
            if isinstance(_this, type) and _this == constraint.table:
                this = constraint.table_name
                if right is None:
                    right = constraint.foreign_table
                break
            elif isinstance(_this, type) and _this == constraint.foreign_table:
                that = constraint.table_name
                if right is None:
                    right = constraint.table
                break
            elif isinstance(_this, SqlAlias) and isinstance(table := _this._value, type) and table == constraint.table:
                this = _this._name
                if right is None:
                    right = constraint.foreign_table
                break
            elif isinstance(_this, SqlAlias) and isinstance(table := _this._value, type) and table == constraint.foreign_table:
                that = _this._name
                if right is None:
                    right = constraint.table
                break
        else:
            raise RuntimeError('improper foreign constraint')

        if isinstance(right, type):
            self.add(table_name(right))
            if this is None and right == constraint.table:
                this = _this.__name__
            if that is None and right == constraint.foreign_table:
                that = _this.__name__
        elif isinstance(right, SqlAlias) and isinstance(table := right._value, type):
            self.add(table_name(table))
            self.add(right._name)

            if this is None and right == constraint.table:
                this = right._name
            if that is None and right == constraint.foreign_table:
                that = right._name
        else:
            raise TypeError(f'JOIN {right}')

        assert this is not None and that is not None

        if constraint.fields == constraint.foreign_fields:
            self.add(['USING', '('])

            for field in constraint.fields:
                self.add(field)
                self.add(',')

            self._stat.pop()
            self.add(')')

        else:
            self.add(['ON', '('])
            for af, bf in zip(constraint.fields, constraint.foreign_fields):
                self.add([f'{this}.{af} = {that}.{bf}'])
                self.add('AND')

            self._stat.pop()
            self.add(')')

        self._involved.append(right)

    @catch_error
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
                self.add(field)
            else:
                raise TypeError('GROUP BY ' + repr(field))
            self.add(',')

        self._stat.pop()
        return self

    @catch_error
    def having(self, *exprs: Union[bool, SqlExpr]) -> Self:
        """
        ``HAVING`` https://www.sqlite.org/lang_select.html#resultset
        """
        if len(exprs) == 0:
            return self

        from .func_stat import and_
        self.add('HAVING')
        self.add(and_(*exprs))

        return self

    @catch_error
    def intersect(self, stat: SqlStat) -> Self:
        """
        ``INTERSECT`` https://www.sqlite.org/lang_select.html#compound_select_statements
        """
        self.add('INTERSECT')
        self._stat.extend(stat._stat)
        stat._connection = None
        stat._stat = None
        return self

    def __and__(self, other: SqlStat) -> Self:
        """
        ``INTERSECT`` https://www.sqlite.org/lang_select.html#compound_select_statements
        """
        return self.intersect(other)

    @catch_error
    def union(self, stat: SqlStat, all=False) -> Self:
        """
        ``UNION`` https://www.sqlite.org/lang_select.html#compound_select_statements
        """
        self.add('UNION')
        if all:
            self.add('ALL')
        self._stat.extend(stat._stat)
        stat._connection = None
        stat._stat = None
        return self

    def __or__(self, other: SqlStat) -> Self:
        """
        ``UNION`` https://www.sqlite.org/lang_select.html#compound_select_statements
        """
        return self.union(other)

    @catch_error
    def except_(self, stat: SqlStat) -> Self:
        """
        ``EXCEPT`` https://www.sqlite.org/lang_select.html#compound_select_statements
        """
        self.add('EXCEPT')
        self._stat.extend(stat._stat)
        stat._connection = None
        stat._stat = None
        return self

    def __sub__(self, other: SqlStat) -> Self:
        """
        ``EXCEPT`` https://www.sqlite.org/lang_select.html#compound_select_statements
        """
        return self.except_(other)


class SqlReturnStat:
    @catch_error
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
                    zelf.add(SqlRemovePlaceHolder(_expr))

                    zelf.add(['AS', exp._name])
                    fields.append(exp._name)
                else:
                    raise TypeError(f'RETURNING ({exp})')

                zelf.add(',')
            zelf._stat.pop()

            if isinstance(self, SqlInsertStat):
                self._return_table = None
                self._fields = fields
            else:
                self.table = None

        return self


class SqlInsertStat(SqlStat[T], SqlReturnStat, Generic[T]):
    def __init__(self, table: type[T], fields: list[str] = None, *, named: bool = False):
        super().__init__(table)
        self._fields = fields
        self._used_fields: list[str] | None = None

        # when
        #   None: `VALUES` set
        #   'DEFAULT': `DEFAULT VALUES` unset
        #   {field->value}: `DEFAULT VALUES` set
        self._values: Union[Literal['DEFAULT'], dict[str, Any], None] = {
            name: SqlLiteral('?')
            for name in table_field_names(table)
        }
        self._named = named

        self._returning = False
        self._return_table = table

    @overload
    def select_from(self, table: type[T], *, distinct: bool = False) -> SqlSelectStat[T]:
        pass

    @overload
    def select_from(self, *field, distinct: bool = False,
                    from_table: Union[str, type, SqlAlias, SqlSelectStat] = None) -> SqlSelectStat[tuple]:
        pass

    @catch_error
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

    @catch_error
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

    @catch_error
    def _set_values(self):
        if isinstance(self._values, str):
            self.add(['DEFAULT', 'VALUES'])
        elif isinstance(self._values, dict):
            self._use_fields = []

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
                    if value.value == '?':
                        if self._named:
                            self.add(f':{field}')
                        else:
                            self.add('?')
                        self._use_fields.append(field)
                    else:
                        self.add(value.value)
                elif isinstance(value, SqlPlaceHolder):
                    self.add(repr(value.value))
                elif isinstance(value, SqlStat):
                    self.add(value)
                elif isinstance(value, SqlExpr):
                    self.add(SqlRemovePlaceHolder(value))
                else:
                    raise TypeError(repr(value))
                self.add(',')
            self._stat.pop()
            self.add(')')

        self._values = None

    @catch_error
    def on_conflict(self, *conflict, where: Union[bool, SqlCompareOper] = None) -> SqlUpsertStat[T]:
        self._set_values()
        return SqlUpsertStat(self, *conflict, where=where)

    @catch_error
    def returning(self, *expr: Union[str, SqlExpr]) -> SqlInsertStat[tuple]:
        self._set_values()
        super().returning(*expr)
        self._returning = True
        return self

    def build(self) -> tuple[str, list]:
        self._set_values()
        return super().build()

    def submit(self, parameter: list[T] = ()) -> Cursor[T]:
        if (connection := self._connection) is None:
            raise RuntimeError('Do not in a connection context')

        self._set_values()

        from .table import table_class
        try:
            table_cls = table_class(self.table)
        except AttributeError:
            pass
        else:
            if self._named:
                def mapper(p):
                    if isinstance(p, self.table):
                        return {f: v for f, v in zip(self._use_fields, table_cls.table_seq(p, self._fields))}
                    return dict(p)
            else:
                def mapper(p):
                    if isinstance(p, self.table):
                        return table_cls.table_seq(p, self._use_fields)
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

        ret = Cursor(connection, cur, self._return_table)
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
                self._stat.add(SqlRemovePlaceHolder(where))
            self._stat.add(')')

    @catch_error(attr='_stat')
    def do_nothing(self) -> SqlInsertStat[T]:
        self._stat.add(['DO', 'NOTHING'])
        return self._stat

    @catch_error(attr='_stat')
    def do_update(self, *args: Union[bool, SqlCompareOper], where: Union[bool, SqlCompareOper] = None) -> SqlInsertStat[T]:
        self._stat.add(['DO', 'UPDATE', 'SET'])

        self._stat.add(SqlRemovePlaceHolder(SqlVarArgOper(',', [
            SqlCompareOper.as_set_expr(it) for it in args
        ])))

        if where is not None:
            self._stat.add('WHERE')
            self._stat.add(SqlRemovePlaceHolder(where))
        return self._stat


class SqlUpdateStat(SqlStat[T], SqlWhereStat, SqlLimitStat, SqlReturnStat, Generic[T]):
    @catch_error
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
