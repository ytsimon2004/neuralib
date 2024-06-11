from __future__ import annotations

import abc
import contextlib
import datetime
from collections.abc import Sequence, Iterable
from typing import overload, Any, TYPE_CHECKING, Literal, ContextManager, TypeVar, Generic, Union, Optional

from typing_extensions import Self

from .table import Field

if TYPE_CHECKING:
    from .stat import SqlStat, SqlSelectStat

__all__ = [
    'SqlExpr',
    'wrap', 'wrap_seq',
    'SqlLiteral', 'SqlPlaceHolder', 'SqlField', 'SqlAlias', 'SqlAliasField', 'SqlSubQuery',
    'SqlOper', 'SqlOrderOper', 'SqlCastOper', 'SqlCompareOper', 'SqlUnaryOper', 'SqlBinaryOper', 'SqlFuncOper',
    'SqlVarArgOper', 'SqlConcatOper', 'SqlExistsOper', 'SqlAggregateFunc', 'SqlWindowDef', 'SqlWindowFunc',
    'SqlCaseExpr', 'SqlCteExpr'
]


class SqlExpr(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __sql_stat__(self, stat: SqlStat):
        pass

    def __matmul__(self, other: str) -> SqlAlias[Self]:
        return SqlAlias(self, other)

    # ============== #
    # equal operator #
    # ============== #

    def __eq__(self, other) -> SqlCompareOper:
        return SqlCompareOper('=', self, wrap(other))

    def __ne__(self, other) -> SqlCompareOper:
        return SqlCompareOper('!=', self, wrap(other))

    def is_null(self) -> SqlCompareOper:
        return SqlCompareOper('IS', self, wrap(None))

    def is_not_null(self) -> SqlCompareOper:
        return SqlCompareOper('IS NOT', self, wrap(None))

    def like(self, value: str) -> SqlCompareOper:
        if value is not None and not isinstance(value, str):
            raise TypeError()

        oper = 'LIKE'
        if value is not None:
            if '%' not in value:
                value += '%'
        return SqlCompareOper(oper, self, SqlLiteral(repr(value)))

    def not_like(self, value: str) -> SqlCompareOper:
        if value is not None and '%' not in value:
            value += '%'
        return SqlCompareOper('NOT LIKE', self, SqlLiteral(repr(value)))

    def glob(self, value: str) -> SqlCompareOper:
        if value is not None and not isinstance(value, str):
            raise TypeError()

        return SqlCompareOper('GLOB', self, SqlLiteral(repr(value)))

    def __contains__(self, item) -> SqlCompareOper:
        from .stat import SqlStat

        if isinstance(item, str):
            return self.like('%' + item + '%')
        elif isinstance(item, (range, slice)):
            return self.between(item)
        elif isinstance(item, (list, tuple, SqlStat)):
            return self.contains(item)
        else:
            raise TypeError()

    @overload
    def between(self, value: Any, value2: Any) -> SqlCompareOper:
        pass

    @overload
    def between(self, value: Union[tuple[Any, Any], range, slice]) -> SqlCompareOper:
        pass

    def between(self, value, value2=None) -> SqlCompareOper:
        if value2 is None:
            return SqlCompareOper('BETWEEN', self, value)
        else:
            return SqlCompareOper('BETWEEN', self, (wrap(value), wrap(value2)))

    @overload
    def not_between(self, value: Any, value2: Any) -> SqlCompareOper:
        pass

    @overload
    def not_between(self, value: Union[tuple[Any, Any], range, slice]) -> SqlCompareOper:
        pass

    def not_between(self, value, value2=None) -> SqlCompareOper:
        if value2 is None:
            return SqlCompareOper('NOT BETWEEN', self, value)
        else:
            return SqlCompareOper('NOT BETWEEN', self, (wrap(value), wrap(value2)))

    def contains(self, value: Union[Sequence, SqlStat]) -> SqlCompareOper:
        from .stat import SqlStat

        if isinstance(value, (tuple, list)):
            return SqlCompareOper('IN', self, tuple(value))
        elif isinstance(value, SqlStat):
            return SqlCompareOper('IN', self, wrap(value))
        else:
            raise TypeError()

    def not_contains(self, value: Union[Sequence, SqlStat]) -> SqlCompareOper:
        from .stat import SqlStat
        if isinstance(value, (tuple, list)):
            return SqlCompareOper('NOT IN', self, tuple(value))
        elif isinstance(value, SqlStat):
            return SqlCompareOper('NOT IN', self, value)
        else:
            raise TypeError()

    # ================ #
    # compare operator #
    # ================ #

    def __lt__(self, other) -> SqlCompareOper:
        return SqlCompareOper('<', self, wrap(other))

    def __le__(self, other) -> SqlCompareOper:
        return SqlCompareOper('<=', self, wrap(other))

    def __gt__(self, other) -> SqlCompareOper:
        return SqlCompareOper('>', self, wrap(other))

    def __ge__(self, other) -> SqlCompareOper:
        return SqlCompareOper('>=', self, wrap(other))

    # =============== #
    # and/or operator #
    # =============== #

    def __and__(self, other: Union[str, SqlExpr]) -> SqlExpr:
        return SqlVarArgOper('AND', [self, wrap(other)])

    def __or__(self, other: Union[str, SqlExpr]) -> SqlExpr:
        return SqlVarArgOper('OR', [self, wrap(other)])

    # =============== #
    # number operator #
    # =============== #

    def __lshift__(self, other: Union[int, SqlExpr]) -> SqlExpr:
        return SqlBinaryOper('<<', self, wrap(other))

    def __rshift__(self, other: Union[int, SqlExpr]) -> SqlExpr:
        return SqlBinaryOper('>>', self, wrap(other))

    def __invert__(self) -> SqlExpr:
        return SqlUnaryOper("~", self)

    def __pos__(self) -> SqlExpr:
        return SqlUnaryOper("+", self)

    def __neg__(self) -> SqlExpr:
        return SqlUnaryOper("-", self)

    def __add__(self, other: Union[int, float, SqlExpr]) -> SqlExpr:
        return SqlBinaryOper("+", self, wrap(other))

    def __radd__(self, other: Union[int, float]) -> SqlExpr:
        return SqlBinaryOper("+", wrap(other), self)

    def __sub__(self, other: Union[int, float, SqlExpr]) -> SqlExpr:
        return SqlBinaryOper("-", self, wrap(other))

    def __rsub__(self, other: Union[int, float]) -> SqlExpr:
        return SqlBinaryOper("-", wrap(other), self)

    def __mul__(self, other: Union[int, float, SqlExpr]) -> SqlExpr:
        return SqlBinaryOper("*", self, wrap(other))

    def __rmul__(self, other: Union[int, float]) -> SqlExpr:
        return SqlBinaryOper("*", wrap(other), self)

    def __truediv__(self, other: Union[int, float, SqlExpr]) -> SqlExpr:
        return SqlBinaryOper("/", self, wrap(other))

    def __rtruediv__(self, other: Union[int, float]) -> SqlExpr:
        return SqlBinaryOper("/", wrap(other), self)

    def __mod__(self, other: Union[int, float, SqlExpr]) -> SqlExpr:
        return SqlBinaryOper('%', self, wrap(other))

    def __rmod__(self, other: Union[int, float]) -> SqlExpr:
        return SqlBinaryOper("%", wrap(other), self)

    # ================= #
    # spectral operator #
    # ================= #

    def cast(self, t: type) -> SqlExpr:
        if t in (bool,):
            return SqlBinaryOper('CAST', self, 'BOOLEAN')
        elif t in (int,):
            return SqlBinaryOper('CAST', self, 'INT')
        elif t in (float,):
            return SqlBinaryOper('CAST', self, 'FLOAT')
        elif t in (str,):
            return SqlBinaryOper('CAST', self, 'TEXT')
        else:
            raise TypeError()


def wrap(other: Any) -> SqlExpr:
    from .stat import SqlStat

    if other is None:
        return SqlLiteral.SQL_NULL
    elif other is True:
        return SqlLiteral.SQL_TRUE
    elif other is False:
        return SqlLiteral.SQL_FALSE
    elif isinstance(other, str) and other.startswith(':'):
        return SqlLiteral(other)
    elif isinstance(other, (int, float, str)):
        return SqlPlaceHolder(other)
    elif isinstance(other, SqlExpr):
        return other
    elif isinstance(other, SqlStat):
        return SqlSubQuery(other)
    elif isinstance(other, datetime.date):
        return SqlLiteral(repr(other.strftime('%Y-%m-%d')))
    elif isinstance(other, datetime.datetime):
        return SqlLiteral(repr(other.strftime('%Y-%m-%d %H:%M:%S')))
    else:
        raise TypeError(repr(other))


def wrap_seq(*other) -> list[SqlExpr]:
    return list(map(wrap, other))


def sql_join(stat: SqlStat, sep: str, exprs: Iterable[SqlExpr]):
    first = True
    for expr in exprs:
        if not first:
            stat.add(sep)
        else:
            first = False

        expr.__sql_stat__(stat)


def sql_join_set(stat: SqlStat, sep: str, exprs: Iterable[Union[bool, SqlCompareOper]]):
    first = True
    for expr in exprs:
        if not first:
            stat.add(sep)
        else:
            first = False

        if isinstance(expr, SqlCompareOper) and expr.oper == '=' and isinstance(expr.left, SqlField):
            field = expr.left.field
            if isinstance(expr.right, SqlPlaceHolder):
                stat.add([field.name, '=', '?'], expr.right.value)
            elif isinstance(expr.right, SqlLiteral):
                stat.add([field.name, '=', '?'], expr.right.value)
            elif isinstance(expr.right, SqlExpr):
                stat.add([field.name, '=', '('])
                expr.right.__sql_stat__(stat)
                stat.add(')')
            else:
                raise TypeError(repr(expr))
        else:
            raise TypeError(repr(expr))


class SqlLiteral(SqlExpr):
    __slots__ = 'value',

    SQL_NULL: 'SqlLiteral'
    SQL_TRUE: 'SqlLiteral'
    SQL_FALSE: 'SqlLiteral'

    def __init__(self, value: str):
        self.value = value

    def __sql_stat__(self, stat: SqlStat):
        stat.add(self.value)

    def __repr__(self):
        return self.value


SqlLiteral.SQL_NULL = SqlLiteral('NULL')
SqlLiteral.SQL_TRUE = SqlLiteral('TRUE')
SqlLiteral.SQL_FALSE = SqlLiteral('FALSE')


class SqlPlaceHolder(SqlExpr):
    __slots__ = 'value',

    def __init__(self, value):
        self.value = value

    def __sql_stat__(self, stat: SqlStat):
        stat.add('?', self.value)

    def __repr__(self):
        return f'?=[{self.value}]'


E = TypeVar('E', bound=SqlExpr)


class SqlAlias(SqlExpr, Generic[E]):
    __slots__ = '_value', '_name'

    def __init__(self, value: Union[E, type[E]], name: str):
        self._value = value
        self._name = name

    def __sql_stat__(self, stat: SqlStat):
        stat.add(self._name)

    def __getattr__(self, item: str) -> SqlAliasField:
        if isinstance(self._value, type):
            from .table import table_field
            table_field(self._value, item)
            return SqlAliasField(self._value, self._name, item)
        elif isinstance(self._value, SqlSubQuery):
            return SqlAliasField(self._value.stat, self._name, item)
        else:
            raise TypeError(f'{self._value}.{item}')


class SqlAliasField(SqlExpr):
    __slots__ = 'table', 'name', 'attr'
    __match_args__ = 'table', 'name', 'attr'

    def __init__(self, table: Union[type, SqlStat, SqlCteExpr], name: str, attr: str):
        self.table = table
        self.name = name
        self.attr = attr

    def __sql_stat__(self, stat: SqlStat):
        stat.add(f'{self.name}.{self.attr}')

    def __str__(self):
        return f'{self.name}.{self.attr}'

    def __repr__(self):
        return f'{self.name}.{self.attr}'


class SqlField(SqlExpr):
    __slots__ = 'field',

    def __init__(self, field: Field):
        self.field = field

    def __sql_stat__(self, stat: SqlStat):
        stat.add(f'{self.table_name}.{self.field.name}')

    @property
    def table(self) -> type:
        return self.field.table

    @property
    def table_name(self) -> str:
        return self.field.table_name

    @property
    def name(self) -> str:
        return self.field.name

    def __repr__(self):
        return f'{self.field.table_name}.{self.field.name}'


class SqlSubQuery(SqlExpr):
    """https://www.sqlite.org/lang_expr.html#subquery_expressions"""

    __slots__ = 'stat',

    def __init__(self, stat: SqlStat):
        self.stat = stat
        stat._connection = None

    def __sql_stat__(self, stat: SqlStat):
        stat.add(self.stat)

    def __repr__(self):
        return '(' + self.stat.build() + ')'


class SqlOper(SqlExpr, metaclass=abc.ABCMeta):
    __slots__ = 'oper',

    def __init__(self, oper: str):
        self.oper = oper


class SqlCompareOper(SqlOper):
    __slots__ = 'oper', 'left', 'right'

    def __init__(self, oper: str, left: SqlExpr, right: Any):
        super().__init__(oper)
        self.left = left
        self.right = right

    def __sql_stat__(self, stat: SqlStat):
        if self.oper in ('IN', 'NOT IN'):
            self.left.__sql_stat__(stat)
            stat.add(self.oper)
            if isinstance(self.right, tuple):
                stat.add(['(', ','.join(map(repr, self.right)), ')'])
            else:
                self.right.__sql_stat__(stat)

        elif self.oper in ('BETWEEN', 'NOT BETWEEN'):
            self.left.__sql_stat__(stat)
            stat.add(self.oper)
            if isinstance(right := self.right, slice):
                a = right.start
                b = right.stop
                if a is None or b is None:
                    raise ValueError()
                stat.add([str(a), 'AND', str(b - 1)])
            elif isinstance(right, range):
                a = right.start
                b = right.stop
                stat.add([str(a), 'AND', str(b - 1)])
            else:
                for i, it in enumerate(right):
                    if i == 1:
                        stat.add('AND')
                    elif i > 1:
                        raise RuntimeError()

                    if isinstance(it, datetime.date):
                        stat.add(it.strftime('%Y-%m-%d'))
                    elif isinstance(it, datetime.datetime):
                        stat.add(it.strftime('%Y-%m-%d %H:%M:%S'))
                    elif isinstance(it, SqlExpr):
                        it.__sql_stat__(stat)
                    else:
                        raise TypeError()
        else:
            self.left.__sql_stat__(stat)
            stat.add(self.oper)
            self.right.__sql_stat__(stat)

    def __repr__(self):
        return '(' + repr(self.left) + self.oper + repr(self.right) + ')'

    def __invert__(self) -> SqlCompareOper:
        return self.not_()

    def not_(self) -> SqlCompareOper:
        if self.oper == '=':
            return SqlCompareOper('!=', self.left, self.right)
        elif self.oper == '!=':
            return SqlCompareOper('=', self.left, self.right)
        elif self.oper == 'LIKE':
            return SqlCompareOper('NOT LIKE', self.left, self.right)
        elif self.oper == 'NOT LIKE':
            return SqlCompareOper('LIKE', self.left, self.right)
        elif self.oper == 'BETWEEN':
            return SqlCompareOper('NOT BETWEEN', self.left, self.right)
        elif self.oper == 'NOT BETWEEN':
            return SqlCompareOper('BETWEEN', self.left, self.right)
        elif self.oper == 'IN':
            return SqlCompareOper('NOT IN', self.left, self.right)
        elif self.oper == 'NOT IN':
            return SqlCompareOper('IN', self.left, self.right)
        elif self.oper == '<':
            return SqlCompareOper('>=', self.left, self.right)
        elif self.oper == '<=':
            return SqlCompareOper('>', self.left, self.right)
        elif self.oper == '>':
            return SqlCompareOper('<=', self.left, self.right)
        elif self.oper == '>=':
            return SqlCompareOper('<', self.left, self.right)
        else:
            raise TypeError(f'{self.oper} not support NOT')


class SqlUnaryOper(SqlOper):
    __slots__ = 'oper', 'right'

    def __init__(self, oper: str, right: SqlExpr):
        super().__init__(oper)
        self.right = right

    def __sql_stat__(self, stat: SqlStat):
        stat.add(self.oper)
        self.right.__sql_stat__(stat)

    def __repr__(self):
        return '(' + self.oper + repr(self.right) + ')'


class SqlCastOper(SqlOper):
    __slots__ = 'oper', 'right'

    def __init__(self, oper: str, right: SqlExpr):
        super().__init__(oper)
        self.right = right

    def __sql_stat__(self, stat: SqlStat):
        stat.add(['CAST', '('])
        self.right.__sql_stat__(stat)
        stat.add(['AS', self.oper, ')'])

    def __repr__(self):
        return 'CAST(' + repr(self.right) + ' AS ' + self.oper + ')'


class SqlOrderOper(SqlOper):
    __slots__ = 'oper', 'right'

    def __init__(self, oper: str, right: SqlExpr):
        super().__init__(oper)
        self.right = right

    def __sql_stat__(self, stat: SqlStat):
        self.right.__sql_stat__(stat)
        stat.add(self.oper)

    def nulls_first(self) -> Self:
        if self.oper in ('ASC', 'DESC'):
            return SqlOrderOper(f'{self.oper} NULLS FIRST', self.right)
        elif self.oper in ('NULLS FIRST', 'ASC NULLS FIRST', 'DESC NULLS FIRST'):
            return self
        else:
            raise RuntimeError(f'ORDER BY {self.oper} NULLS FIRST')

    def nulls_last(self) -> Self:
        if self.oper in ('ASC', 'DESC'):
            return SqlOrderOper(f'{self.oper} NULLS LAST', self.right)
        elif self.oper in ('NULLS LAST', 'ASC NULLS LAST', 'DESC NULLS LAST'):
            return self
        else:
            raise RuntimeError(f'ORDER BY {self.oper} NULLS LAST')

    def __repr__(self):
        return repr(self.right) + ' ' + self.oper


class SqlBinaryOper(SqlOper):
    __slots__ = 'oper', 'left', 'right'

    def __init__(self, oper: str, left: SqlExpr, right: Union[str, SqlExpr]):
        super().__init__(oper)
        self.left = left
        self.right = right

    def __sql_stat__(self, stat: SqlStat):
        self.left.__sql_stat__(stat)
        stat.add(self.oper)
        self.right.__sql_stat__(stat)

    def __repr__(self):
        return '(' + repr(self.left) + self.oper + repr(self.right) + ')'


class SqlVarArgOper(SqlOper):
    __slots__ = 'oper', 'args'

    def __init__(self, oper: str, args: list[SqlExpr]):
        super().__init__(oper)
        self.args = args

    def __sql_stat__(self, stat: SqlStat):
        if self.oper in ('NOT AND', 'NOT OR'):
            stat.add(['NOT', '('])
            self.not_().__sql_stat__(stat)
            stat.add(')')
        elif self.oper in ('AND', 'OR'):
            stat.add('(')
            sql_join(stat, self.oper, self.args)
            stat.add(')')
        else:
            raise ValueError()

    def __repr__(self):
        return self.oper + '(' + ','.join(map(repr, self.args)) + ')'

    def __invert__(self) -> SqlVarArgOper:
        return self.not_()

    def not_(self) -> SqlVarArgOper:
        if self.oper == 'AND':
            return SqlVarArgOper('NOT AND', self.args)
        elif self.oper == 'OR':
            return SqlVarArgOper('NOT OR', self.args)
        elif self.oper == 'NOT AND':
            return SqlVarArgOper('AND', self.args)
        elif self.oper == 'NOT OR':
            return SqlVarArgOper('OR', self.args)
        else:
            raise TypeError(f'NOT {self.oper}')

    def __and__(self, other: Union[str, SqlExpr]) -> SqlVarArgOper:
        if self.oper == 'AND':
            return SqlVarArgOper('AND', [*self.args, wrap(other)])
        return SqlVarArgOper('AND', [self, wrap(other)])

    def __or__(self, other: Union[str, SqlExpr]) -> SqlVarArgOper:
        if self.oper == 'OR':
            return SqlVarArgOper('OR', [*self.args, wrap(other)])
        return SqlVarArgOper('OR', [self, wrap(other)])


class SqlConcatOper(SqlOper):
    __slots__ = 'args',

    def __init__(self, args: list[SqlExpr], oper='||'):
        super().__init__(oper)
        self.oper = oper
        self.args = args

    def __sql_stat__(self, stat: SqlStat):
        sql_join(stat, self.oper, self.args)

    def __repr__(self):
        return '||'.join(map(repr, self.args))


class SqlFuncOper(SqlOper):
    __slots__ = 'oper', 'args'

    def __init__(self, oper: str, *args):
        super().__init__(oper)
        self.args = [wrap(it) for it in args]

    def __sql_stat__(self, stat: SqlStat):
        stat.add(self.oper)

        if len(self.args):
            stat.add('(')
            sql_join(stat, ',', self.args)
            stat.add(')')
        else:
            stat.add('()')

    def __repr__(self):
        return self.oper + '(' + ','.join(map(repr, self.args)) + ')'


class SqlExistsOper(SqlOper):
    """https://www.sqlite.org/lang_expr.html#the_exists_operator"""
    __slots__ = 'oper', 'stat'

    def __init__(self, oper: Literal['EXISTS', 'NOT EXISTS'], stat: SqlStat):
        super().__init__(oper)
        self.stat = stat

    def __sql_stat__(self, stat: SqlStat):
        stat.add(self.oper)
        stat.add(self.stat)

    def __invert__(self):
        if self.oper == 'EXISTS':
            return SqlExistsOper('NOT EXISTS', self.stat)
        elif self.oper == 'NOT EXISTS':
            return SqlExistsOper('EXISTS', self.stat)
        else:
            raise RuntimeError()


class SqlAggregateFunc(SqlFuncOper):

    def __init__(self, oper: str, *args):
        super().__init__(oper, *args)
        self._where: Optional[SqlVarArgOper] = None
        self._distinct = False

    def __sql_stat__(self, stat: SqlStat):
        stat.add(self.oper)

        stat.add('(')
        if self._distinct:
            stat.add('DISTINCT')

        if len(self.args):
            sql_join(stat, ',', self.args)
        stat.add(')')

        if (where := self._where) is not None:  # type: SqlVarArgOper
            stat.add(['FILTER', '(', 'WHERE'])
            where.__sql_stat__(stat)
            stat.add(')')

    def distinct(self) -> Self:
        self._distinct = True
        return self

    def where(self, *args: Union[bool, SqlExpr]) -> Self:
        self._where = SqlVarArgOper('AND', list(map(wrap, args)))
        return self


class SqlWindowDef(SqlExpr):

    def __init__(self, name: str = None):
        self.name = name
        self._partition: Optional[list[SqlExpr]] = None
        self._order_by: Optional[list[SqlExpr]] = None
        self._frame_spec: SqlWindowFrame = None

    def __sql_stat__(self, stat: SqlStat):
        stat.add('(')
        if (p := self._partition) is not None:
            stat.add('PARTITION BY')
            sql_join(stat, ',', p)

        if (p := self._order_by) is not None:
            stat.add('ORDER BY')
            sql_join(stat, ',', p)

        if (spec := self._frame_spec) is not None:
            spec.__sql_stat__(stat)

        stat.add(')')

    def over(self, *, order_by=None, partition_by=None) -> Self:
        if isinstance(order_by, (tuple, list)):
            self._order_by = wrap_seq(*order_by)
        elif order_by is not None:
            self._order_by = [wrap(order_by)]

        if isinstance(partition_by, (tuple, list)):
            self._partition = wrap_seq(*partition_by)
        elif partition_by is not None:
            self._partition = [wrap(partition_by)]

        return self

    @overload
    def frame(self, on: Literal['RANGE', 'ROWS', 'GROUPS']) -> ContextManager[SqlWindowFrame]:
        pass

    @contextlib.contextmanager
    def frame(self, on):
        frame = SqlWindowFrame(on)
        yield frame
        self._frame_spec = frame


class SqlWindowFrame:
    class Spec:
        __slots__ = 'kind', 'expr'
        __match_args__ = 'kind', 'expr'

        def __init__(self, kind: Literal['UNBOUNDED PRECEDING', 'CURRENT ROW', 'PRECEDING', 'FOLLOWING'],
                     expr: SqlExpr = None):
            self.kind = kind
            self.expr = expr

        def __sql_stat__(self, stat: SqlStat):
            if self.kind in ('UNBOUNDED PRECEDING', 'CURRENT ROW'):
                stat.add(self.kind)
            elif self.kind in ('PRECEDING', 'FOLLOWING'):
                self.expr.__sql_stat__(stat)
                stat.add(self.kind)
            else:
                raise ValueError()

    def __init__(self, on: Literal['RANGE', 'ROWS', 'GROUPS']):
        self._on = on
        self._specs = []
        self._exclude = None

    def __sql_stat__(self, stat: SqlStat):
        stat.add(self._on)

        spec = self._specs[-1]
        if isinstance(spec, SqlWindowFrame.Spec) and spec.kind in ('UNBOUNDED PRECEDING', 'CURRENT ROW', 'PRECEDING'):
            spec.__sql_stat__(stat)
        elif isinstance(spec, tuple) and len(spec) == 2 and isinstance(spec[0], SqlWindowFrame.Spec) and isinstance(
                spec[1], SqlWindowFrame.Spec):
            stat.add('BETWEEN')
            spec[0].__sql_stat__(stat)
            stat.add('AND')
            spec[1].__sql_stat__(stat)
        else:
            raise TypeError()

        if (ex := self._exclude) is not None:
            stat.add(['EXCLUDE', ex])

    # noinspection PyUnusedLocal
    def between(self, left, right) -> Self:
        *_, left, right = self._specs
        self._specs.append((left, right))
        return self

    def unbounded_preceding(self) -> Self:
        ret = self.Spec('UNBOUNDED PRECEDING')
        self._specs.append(ret)
        return self

    def current_row(self) -> Self:
        ret = self.Spec('CURRENT ROW')
        self._specs.append(ret)
        return self

    def preceding(self, expr) -> Self:
        ret = self.Spec('PRECEDING', wrap(expr))
        self._specs.append(ret)
        return self

    def following(self, expr) -> Self:
        ret = self.Spec('FOLLOWING', wrap(expr))
        self._specs.append(ret)
        return self

    def exclude(self, on: Literal['NO OTHERS', 'CURRENT ROW', 'GROUP', 'TIES']):
        self._exclude = on


class SqlWindowFunc(SqlAggregateFunc):

    def __init__(self, oper: str, *args):
        SqlAggregateFunc.__init__(self, oper, *args)
        self._over: Union[str, SqlWindowDef, None] = None

    def __sql_stat__(self, stat: SqlStat):
        super().__sql_stat__(stat)

        if self._over is not None:
            stat.add('OVER')
            if isinstance(self._over, str):
                stat.add(self._over)
            elif isinstance(self._over, SqlWindowDef):
                if self._over.name is None:
                    self._over.__sql_stat__(stat)
                elif isinstance(self._over.name, str):
                    stat.add(self._over.name)
                    stat.window_def(self._over)
                else:
                    raise TypeError()
            elif isinstance(self._over, SqlAlias) and isinstance(self._over._value, SqlWindowDef):
                stat.add(self._over._name)
                stat.window_def(self._over._value)
            else:
                raise TypeError()

    @overload
    def over(self, name: Union[str, SqlWindowDef, SqlAlias[SqlWindowDef]]) -> Self:
        pass

    @overload
    def over(self, *, order_by=None, partition_by=None) -> Self:
        pass

    def over(self, name: str = None, *, order_by=None, partition_by=None) -> Self:
        if name is None:
            self._over = over = SqlWindowDef()
            over.over(order_by=order_by, partition_by=partition_by)
        elif isinstance(name, str):
            self._over = name
        elif isinstance(name, SqlWindowDef):
            self._over = name
        elif isinstance(name, SqlAlias) and isinstance(name._value, SqlWindowDef):
            self._over = name
        else:
            raise TypeError()

        return self


class SqlCaseExpr(SqlExpr):
    """https://www.sqlite.org/lang_expr.html#the_case_expression"""

    __slots__ = 'expr', 'cases'
    __match_args__ = 'expr', 'cases'

    def __init__(self, expr: SqlExpr = None):
        self.expr: Optional[SqlExpr] = expr
        self.cases: list[tuple[Optional[SqlExpr], SqlExpr]] = []

    def __sql_stat__(self, stat: SqlStat):
        stat.add('CASE')
        if self.expr is not None:
            self.expr.__sql_stat__(stat)

        for case, then in self.cases:  # type: SqlExpr , SqlExpr
            if case is not None:
                stat.add('WHEN')
                case.__sql_stat__(stat)
                stat.add('THEN')
            else:
                stat.add('ELSE')
            then.__sql_stat__(stat)

        stat.add('END')

    def when(self, case, then) -> Self:
        if len(self.cases) and self.cases[-1][0] is None:
            raise RuntimeError('when after else')

        if isinstance(case, str):
            case = SqlLiteral(repr(case))
        else:
            case = wrap(case)

        if isinstance(then, str):
            then = SqlLiteral(repr(then))
        else:
            then = wrap(then)

        self.cases.append((case, then))
        return self

    def else_(self, then) -> Self:
        if len(self.cases) == 0:
            raise RuntimeError('else without when')
        elif len(self.cases) and self.cases[-1][0] is None:
            raise RuntimeError('else after else')

        if isinstance(then, str):
            then = SqlLiteral(repr(then))
        else:
            then = wrap(then)

        self.cases.append((None, then))
        return self


class SqlCteExpr(SqlExpr):
    __slots__ = '_name', '_select'

    def __init__(self, name: str, select: SqlSelectStat):
        self._name = name
        self._select = select
        select._connection = None

    def __getattr__(self, item) -> SqlAliasField:
        return SqlAliasField(self, self._name, item)

    def __sql_stat__(self, stat: SqlStat):
        stat.add(['WITH', self._name, 'AS'])
        stat.add(self._select)
