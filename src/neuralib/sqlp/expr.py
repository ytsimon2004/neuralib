from __future__ import annotations

import abc
import contextlib
import datetime
from collections.abc import Sequence, Iterable
from typing import overload, Any, TYPE_CHECKING, Literal, ContextManager, TypeVar, Generic, Union, Optional

from typing_extensions import Self

from .annotation import CURRENT_TIMESTAMP, CURRENT_DATE, CURRENT_TIME
from .table import Field

if TYPE_CHECKING:
    from .stat import SqlStat, SqlSelectStat

__all__ = [
    'SqlExpr',
    'wrap', 'wrap_seq', 'eval_expr', 'use_table_first', 'use_table',
    'SqlLiteral', 'SqlPlaceHolder', 'SqlField', 'SqlAlias', 'SqlAliasField', 'SqlSubQuery',
    'SqlOper', 'SqlOrderOper', 'SqlCastOper', 'SqlCompareOper', 'SqlUnaryOper', 'SqlBinaryOper', 'SqlFuncOper',
    'SqlVarArgOper', 'SqlConcatOper', 'SqlExistsOper', 'SqlAggregateFunc', 'SqlWindowDef', 'SqlWindowFunc',
    'SqlCaseExpr', 'SqlCteExpr'
]


class SqlStatBuilder:
    def __init__(self):
        self.stat = []
        self.para = []
        self._deparameter: bool | Literal['repr'] = False

    def build(self) -> str:
        """build a SQL statement."""
        return ' '.join(self.stat)

    def add(self, stat: str | list[str] | SqlStat) -> Self:
        """
        Add SQL token.
        """
        from .stat import SqlStat

        if isinstance(stat, SqlStat):
            is_init_stat = len(self.stat) == 0
            if not is_init_stat:
                self.stat.append('(')

            elements = list(stat._stat)
            if is_init_stat:
                while len(elements) and isinstance(elements[0], SqlCteExpr):
                    if len(self.stat) == 0:
                        self.stat.append('WITH')
                    else:
                        self.stat.append(',')
                    elements.pop(0).__sql_stat__(self)

            for element in elements:
                self.add(element)

            if not is_init_stat:
                self.stat.append(')')

        elif isinstance(stat, str):
            self.stat.append(stat)

        elif isinstance(stat, (tuple, list)):
            self.stat.extend(stat)

        elif isinstance(stat, SqlExpr):
            stat.__sql_stat__(self)

        else:
            raise TypeError(f'{type(stat).__name__}: {repr(stat)}')

        return self


class SqlExpr(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def __sql_stat__(self, stat: SqlStatBuilder):
        pass

    @abc.abstractmethod
    def __sql_eval__(self, data):
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
        return SqlCompareOper('IS', self, SqlLiteral.SQL_NULL)

    def is_not_null(self) -> SqlCompareOper:
        return SqlCompareOper('IS NOT', self, SqlLiteral.SQL_NULL)

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
            return SqlCastOper('BOOLEAN', self)
        elif t in (int,):
            return SqlCastOper('INT', self)
        elif t in (float,):
            return SqlCastOper('FLOAT', self)
        elif t in (str,):
            return SqlCastOper('TEXT', self)
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
    elif other is CURRENT_TIMESTAMP:
        return SqlLiteral('CURRENT_TIMESTAMP')
    elif other is CURRENT_TIME:
        return SqlLiteral('CURRENT_TIME')
    elif other is CURRENT_DATE:
        return SqlLiteral('CURRENT_DATE')
    elif isinstance(other, str) and other.startswith(':'):
        return SqlLiteral(other)
    elif isinstance(other, (int, float)):
        return SqlLiteral(repr(other))
    elif isinstance(other, str):
        return SqlPlaceHolder(other)
    elif isinstance(other, Field):
        return SqlField(other)
    elif isinstance(other, SqlExpr):
        return other
    elif isinstance(other, SqlStat):
        return SqlSubQuery(other)
    elif isinstance(other, datetime.datetime):
        return SqlLiteral(repr(other.strftime('%Y-%m-%d %H:%M:%S.%f')))
    elif isinstance(other, datetime.date):
        return SqlLiteral(repr(other.strftime('%Y-%m-%d')))
    else:
        # delay error raising
        return SqlError(TypeError(repr(other)))


def wrap_seq(*other) -> list[SqlExpr]:
    return list(map(wrap, other))


def eval_expr(expr: SqlExpr | Any, data) -> Any:
    return expr.__sql_eval__(data)


def eval_expr_seq(expr: list[SqlExpr | Any], data) -> list[Any]:
    return [it.__sql_eval__(data) for it in expr]


def sql_join(stat: SqlStatBuilder, sep: str, exprs: Iterable[SqlExpr]):
    first = True
    for expr in exprs:
        if not first:
            stat.add(sep)
        else:
            first = False

        expr.__sql_stat__(stat)


def use_table_first(expr: SqlExpr) -> type | None:
    if isinstance(expr, SqlField):
        return expr.table
    elif isinstance(expr, SqlAlias) and isinstance(table := expr._value, type):
        return table
    elif isinstance(expr, SqlAlias) and isinstance(expr := expr._value, SqlExpr):
        return use_table_first(expr)
    elif isinstance(expr, SqlAliasField) and isinstance(table := expr.table, type):
        return table
    elif isinstance(expr, SqlExistsOper):
        return expr.stat.table
    elif isinstance(expr, (SqlCompareOper, SqlBinaryOper)):
        return use_table_first(expr.left) or use_table_first(expr.right)
    elif isinstance(expr, (SqlUnaryOper, SqlCastOper)):
        return use_table_first(expr.right)
    elif isinstance(expr, (SqlVarArgOper, SqlConcatOper, SqlFuncOper)):
        for arg in expr.args:
            if (ret := use_table_first(arg)) is not None:
                return ret
    return None


def use_table(expr: SqlExpr, self: list[type | SqlAlias]) -> type | SqlAlias[type] | None:
    use_self = []
    for t in self:
        if isinstance(t, type):
            use_self.append(t)
        elif isinstance(t, SqlAlias) and isinstance(table := t._value, type):
            use_self.append(table)

    return _use_table(expr, use_self)


def _use_table(expr: SqlExpr, self: list[type]) -> type | SqlAlias[type] | None:
    if isinstance(expr, SqlField) and isinstance(field := expr.field, Field):
        table = field.table
        return table if table not in self else None
    elif isinstance(expr, SqlAlias) and isinstance(table := expr._value, type):
        return expr if table not in self else None
    elif isinstance(expr, SqlAliasField) and isinstance(table := expr.table, type):
        name = expr.name
        return SqlAlias(table, name) if table not in self else None
    elif isinstance(expr, SqlAlias) and isinstance(expr := expr._value, SqlExpr):
        return _use_table(expr, self)
    elif isinstance(expr, (SqlCompareOper, SqlBinaryOper)):
        return _use_table(expr.left, self) or _use_table(expr.right, self)
    elif isinstance(expr, (SqlUnaryOper, SqlCastOper)):
        return _use_table(expr.right, self)
    elif isinstance(expr, (SqlVarArgOper, SqlConcatOper, SqlFuncOper)):
        for arg in expr.args:
            if (ret := _use_table(arg, self)) is not None:
                return ret
    return None


class SqlError(SqlExpr):
    __slots__ = 'error',

    def __init__(self, error: BaseException):
        self.error = error

    def __sql_stat__(self, stat: SqlStatBuilder):
        raise self.error

    def __sql_eval__(self, data):
        raise self.error

    def __repr__(self):
        return repr(self.error)


class SqlLiteral(SqlExpr):
    __slots__ = 'value',

    SQL_NULL: 'SqlLiteral'
    SQL_TRUE: 'SqlLiteral'
    SQL_FALSE: 'SqlLiteral'

    def __init__(self, value: str):
        self.value = value

    def __sql_stat__(self, stat: SqlStatBuilder):
        stat.add(self.value)

    def __sql_eval__(self, data):
        if self.value == 'NULL':
            return None
        elif self.value == 'TRUE':
            return True
        elif self.value == 'FALSE':
            return False
        else:
            return eval(self.value)

    def __repr__(self):
        return self.value


SqlLiteral.SQL_NULL = SqlLiteral('NULL')
SqlLiteral.SQL_TRUE = SqlLiteral('TRUE')
SqlLiteral.SQL_FALSE = SqlLiteral('FALSE')


class SqlPlaceHolder(SqlExpr):
    __slots__ = 'value',

    def __init__(self, value):
        self.value = value

    def __sql_stat__(self, stat: SqlStatBuilder):
        if isinstance(self.value, SqlError):
            self.value.__sql_stat__(stat)

        if stat._deparameter is True:
            stat.add(repr(self.value))
        elif stat._deparameter == 'repr':
            stat.add(f'?({repr(self.value)})')
        else:
            stat.add('?')
            stat.para.append(self.value)

    def __sql_eval__(self, data):
        return self.value

    def __repr__(self):
        return f'?=[{self.value}]'


class SqlRemovePlaceHolder(SqlExpr):
    __slots__ = 'expr', 'mode'

    def __init__(self, expr: SqlExpr, *, mode: bool | Literal['repr'] = True):
        self.expr = expr
        self.mode = mode

    def __sql_stat__(self, stat: SqlStatBuilder):
        old = stat._deparameter
        stat._deparameter = self.mode
        self.expr.__sql_stat__(stat)
        stat._deparameter = old

    def __sql_eval__(self, data):
        return self.expr.__sql_eval__(data)

    def __repr__(self):
        return repr(self.expr)


E = TypeVar('E')


class SqlAlias(SqlExpr, Generic[E]):
    __slots__ = '_value', '_name'

    def __init__(self, value: Union[E, type[E]], name: str):
        self._value = value
        self._name = name

    def __sql_stat__(self, stat: SqlStatBuilder):
        stat.add(self._name)

    def __sql_eval__(self, data):
        raise NotImplementedError(repr(self))

    def __getattr__(self, item: str) -> SqlAliasField:
        if isinstance(self._value, type):
            from .table import table_field
            table_field(self._value, item)
            return SqlAliasField(self._value, self._name, item)
        elif isinstance(self._value, SqlSubQuery):
            return SqlAliasField(self._value.stat, self._name, item)
        else:
            raise TypeError(f'{self._value}.{item}')

    def __str__(self):
        if isinstance(self._value, type):
            from .table import table_name
            return f"Alias[{table_name(self._value)} AS {self._name}]"
        else:
            return f"Alias[... AS {self._name}]"

    __repr__ = __str__


class SqlAliasField(SqlExpr):
    __slots__ = 'table', 'name', 'attr'

    def __init__(self, table: Union[type, SqlStat, SqlCteExpr], name: str, attr: str):
        self.table = table
        self.name = name
        self.attr = attr

    def __sql_stat__(self, stat: SqlStatBuilder):
        stat.add(f'{self.name}.{self.attr}')

    def __sql_eval__(self, data):
        raise NotImplementedError(repr(self))

    def __str__(self):
        return f'{self.name}.{self.attr}'

    __repr__ = __str__


class SqlField(SqlExpr):
    __slots__ = 'field',

    def __init__(self, field: Field):
        self.field = field

    def __sql_stat__(self, stat: SqlStatBuilder):
        stat.add(f'{self.table_name}.{self.field.name}')

    def __sql_eval__(self, data):
        return getattr(data, self.field.name)

    @property
    def table(self) -> type:
        return self.field.table

    @property
    def table_name(self) -> str:
        return self.field.table_name

    @property
    def name(self) -> str:
        return self.field.name

    def __str__(self):
        return f'{self.field.table_name}.{self.field.name}'

    __repr__ = __str__


class SqlSubQuery(SqlExpr):
    """https://www.sqlite.org/lang_expr.html#subquery_expressions"""

    __slots__ = 'stat',

    def __init__(self, stat: SqlStat):
        self.stat = stat
        stat._connection = None

    def __sql_stat__(self, stat: SqlStatBuilder):
        stat.add(self.stat)

    def __sql_eval__(self, data):
        raise NotImplementedError(repr(self))

    def __str__(self):
        return '(...)'

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

    def __sql_stat__(self, stat: SqlStatBuilder):
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

    def __sql_eval__(self, data):
        if self.oper == '=':
            return self.left.__sql_eval__(data) == self.right.__sql_eval__(data)
        elif self.oper == '!=':
            return self.left.__sql_eval__(data) != self.right.__sql_eval__(data)
        elif self.oper == '<':
            return self.left.__sql_eval__(data) < self.right.__sql_eval__(data)
        elif self.oper == '<=':
            return self.left.__sql_eval__(data) <= self.right.__sql_eval__(data)
        elif self.oper == '>':
            return self.left.__sql_eval__(data) > self.right.__sql_eval__(data)
        elif self.oper == '>=':
            return self.left.__sql_eval__(data) >= self.right.__sql_eval__(data)
        elif self.oper == 'IS':
            return self.left.__sql_eval__(data) is self.right.__sql_eval__(data)
        elif self.oper == 'IS NOT':
            return self.left.__sql_eval__(data) is not self.right.__sql_eval__(data)
        elif self.oper in ('LIKE', 'NOT LIKE'):
            import re
            if not isinstance(text := self.left.__sql_eval__(data), str):
                raise TypeError()
            if not isinstance(pattern := self.right.__sql_eval__(data), str):
                raise TypeError()

            m = re.compile(pattern.replace('%', '.*')).match(text)
            return (self.oper == 'LIKE') == (m is not None)
        elif self.oper == 'GLOB':
            raise NotImplementedError(repr(self))
        elif self.oper in ('BETWEEN', 'NOT BETWEEN'):
            value = self.left.__sql_eval__(data)

            if isinstance(self.right, slice) and (self.right.start is None or self.right.stop is None):
                raise ValueError()
            elif isinstance(self.right, (slice, range)):
                result = self.right.start <= value < self.right.stop
            elif isinstance(self.right, tuple):
                a, b = self.right
                if isinstance(a, (datetime.date, datetime.datetime)) and isinstance(b, (datetime.date, datetime.datetime)):
                    result = a <= value <= b
                elif isinstance(a, SqlExpr) and isinstance(b, SqlExpr):
                    a = a.__sql_eval__(data)
                    b = b.__sql_eval__(data)
                    result = a <= value <= b
                else:
                    raise TypeError()

            return (self.oper == 'BETWEEN') == result

        elif self.oper in ('IN', 'NOT IN'):
            value = self.left.__sql_eval__(data)

            coll = self.right
            if isinstance(coll, SqlExpr):
                coll = self.right.__sql_eval__(data)

            return (self.oper == 'IN') == (value in coll)
        else:
            raise NotImplementedError(repr(self))

    def __str__(self):
        return '(' + str(self.left) + ' ' + self.oper + ' ' + str(self.right) + ')'

    def __repr__(self):
        return '(' + repr(self.left) + ' ' + self.oper + ' ' + repr(self.right) + ')'

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

    @classmethod
    def as_set_expr(cls, expr: SqlCompareOper) -> SqlCompareOper:
        if isinstance(expr, SqlCompareOper) and expr.oper == '=' and isinstance(field := expr.left, SqlField):
            return SqlLiteral(field.name) == expr.right
        raise TypeError(repr(expr))


class SqlUnaryOper(SqlOper):
    __slots__ = 'oper', 'right'

    def __init__(self, oper: str, right: SqlExpr):
        super().__init__(oper)
        self.right = right

    def __sql_stat__(self, stat: SqlStatBuilder):
        stat.add(self.oper)
        self.right.__sql_stat__(stat)

    def __sql_eval__(self, data):
        if self.oper == '~':
            return not (self.right.__sql_eval__(data))
        elif self.oper == '+':
            return +(self.right.__sql_eval__(data))
        elif self.oper == '-':
            return -(self.right.__sql_eval__(data))
        else:
            raise NotImplementedError(repr(self))

    def __repr__(self):
        return '(' + self.oper + repr(self.right) + ')'


class SqlCastOper(SqlOper):
    __slots__ = 'oper', 'right'

    def __init__(self, oper: str, right: SqlExpr):
        super().__init__(oper)
        self.right = right

    def __sql_stat__(self, stat: SqlStatBuilder):
        stat.add(['CAST', '('])
        self.right.__sql_stat__(stat)
        stat.add(['AS', self.oper, ')'])

    def __sql_eval__(self, data):
        if self.oper == 'BOOLEAN':
            return bool(data)
        elif self.oper == 'INT':
            return int(data)
        elif self.oper == 'FLOAT':
            return float(data)
        elif self.oper == 'TEXT':
            return str(data)
        else:
            raise NotImplementedError(repr(self))

    def __repr__(self):
        return 'CAST(' + repr(self.right) + ' AS ' + self.oper + ')'


class SqlOrderOper(SqlOper):
    __slots__ = 'oper', 'right'

    def __init__(self, oper: str, right: SqlExpr):
        super().__init__(oper)
        self.right = right

    def __sql_stat__(self, stat: SqlStatBuilder):
        self.right.__sql_stat__(stat)
        stat.add(self.oper)

    def __sql_eval__(self, data):
        raise NotImplementedError(repr(self))

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

    def __sql_stat__(self, stat: SqlStatBuilder):
        self.left.__sql_stat__(stat)
        stat.add(self.oper)
        self.right.__sql_stat__(stat)

    def __sql_eval__(self, data):
        if self.oper == '+':
            return self.left.__sql_eval__(data) + self.right.__sql_eval__(data)
        elif self.oper == '-':
            return self.left.__sql_eval__(data) - self.right.__sql_eval__(data)
        elif self.oper == '*':
            return self.left.__sql_eval__(data) * self.right.__sql_eval__(data)
        elif self.oper == '/':
            return self.left.__sql_eval__(data) / self.right.__sql_eval__(data)
        elif self.oper == '%':
            return self.left.__sql_eval__(data) % self.right.__sql_eval__(data)
        elif self.oper == '<<':
            return self.left.__sql_eval__(data) << self.right.__sql_eval__(data)
        elif self.oper == '>>':
            return self.left.__sql_eval__(data) >> self.right.__sql_eval__(data)
        else:
            raise NotImplementedError(repr(self))

    def __repr__(self):
        return '(' + repr(self.left) + self.oper + repr(self.right) + ')'


class SqlVarArgOper(SqlOper):
    __slots__ = 'oper', 'args'

    def __init__(self, oper: str, args: list[SqlExpr]):
        super().__init__(oper)
        self.args = args

    def __sql_stat__(self, stat: SqlStatBuilder):
        if self.oper in ('NOT AND', 'NOT OR'):
            stat.add(['NOT', '('])
            self.not_().__sql_stat__(stat)
            stat.add(')')
        elif self.oper in ('AND', 'OR'):
            stat.add('(')
            sql_join(stat, self.oper, self.args)
            stat.add(')')
        elif self.oper == ',':
            sql_join(stat, self.oper, self.args)
        else:
            raise ValueError(f'oper={self.oper}')

    def __sql_eval__(self, data):
        raise NotImplementedError(repr(self))

    def __repr__(self):
        if self.oper == ',':
            return ','.join(map(repr, self.args))
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

    def __sql_stat__(self, stat: SqlStatBuilder):
        sql_join(stat, self.oper, self.args)

    def __sql_eval__(self, data):
        return ''.join(map(str, eval_expr_seq(self.args, data)))

    def __repr__(self):
        return '||'.join(map(repr, self.args))


class SqlFuncOper(SqlOper):
    __slots__ = 'oper', 'args'

    def __init__(self, oper: str, func, *args):
        super().__init__(oper)
        self.args = [wrap(it) for it in args]
        self.func = func

    def __sql_stat__(self, stat: SqlStatBuilder):
        stat.add(self.oper)

        if len(self.args):
            stat.add('(')
            sql_join(stat, ',', self.args)
            stat.add(')')
        else:
            stat.add('()')

    def __sql_eval__(self, data):
        if self.func is None:
            raise NotImplementedError()

        return self.func(*eval_expr_seq(self.args, data))

    def __repr__(self):
        return self.oper + '(' + ','.join(map(repr, self.args)) + ')'


class SqlExistsOper(SqlOper):
    """https://www.sqlite.org/lang_expr.html#the_exists_operator"""
    __slots__ = 'oper', 'stat'

    def __init__(self, oper: Literal['EXISTS', 'NOT EXISTS'], stat: SqlStat):
        super().__init__(oper)
        self.stat = stat
        stat._connection = None

    def __sql_stat__(self, stat: SqlStatBuilder):
        stat.add(self.oper)
        stat.add(self.stat)

    def __sql_eval__(self, data):
        raise NotImplementedError(repr(self))

    def __invert__(self):
        if self.oper == 'EXISTS':
            return SqlExistsOper('NOT EXISTS', self.stat)
        elif self.oper == 'NOT EXISTS':
            return SqlExistsOper('EXISTS', self.stat)
        else:
            raise RuntimeError()


class SqlAggregateFunc(SqlFuncOper):

    def __init__(self, oper: str, *args):
        super().__init__(oper, None, *args)
        self._where: Optional[SqlVarArgOper] = None
        self._distinct = False

    def __sql_stat__(self, stat: SqlStatBuilder):
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

    def __sql_eval__(self, data):
        raise NotImplementedError(repr(self))

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

    def __sql_stat__(self, stat: SqlStatBuilder):
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

    def __sql_eval__(self, data):
        raise NotImplementedError(repr(self))

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

        def __init__(self, kind: Literal['UNBOUNDED PRECEDING', 'CURRENT ROW', 'PRECEDING', 'FOLLOWING'],
                     expr: SqlExpr = None):
            self.kind = kind
            self.expr = expr

        def __sql_stat__(self, stat: SqlStatBuilder):
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

    def __sql_stat__(self, stat: SqlStatBuilder):
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

    def __sql_stat__(self, stat: SqlStatBuilder):
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
                else:
                    raise TypeError()
            elif isinstance(self._over, SqlAlias) and isinstance(self._over._value, SqlWindowDef):
                stat.add(self._over._name)
            else:
                raise TypeError()

    def __sql_eval__(self, data):
        raise NotImplementedError(repr(self))

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

    def __init__(self, expr: SqlExpr = None):
        self.expr: Optional[SqlExpr] = expr
        self.cases: list[tuple[Optional[SqlExpr], SqlExpr]] = []

    def __sql_stat__(self, stat: SqlStatBuilder):
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

    def __sql_eval__(self, data):
        raise NotImplementedError(repr(self))

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

    def __sql_stat__(self, stat: SqlStatBuilder):
        stat.add([self._name, 'AS'])
        stat.add(self._select)

    def __sql_eval__(self, data):
        raise NotImplementedError(repr(self))
