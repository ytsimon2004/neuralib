"""
SQL functions.
"""
from __future__ import annotations

from typing import overload

from . import expr
from .func_dec import as_func_expr

__all__ = [
    'abs', 'char', 'coalesce', 'hex', 'ifnull', 'iff', 'instr', 'length', 'likelihood', 'lower', 'ltrim', 'max',
    'min', 'nullif', 'quote', 'random', 'randomblob', 'replace', 'round', 'rtrim', 'sign', 'substr', 'trim', 'typeof',
    'unhex', 'unicode', 'unlikely', 'upper', 'zeroblob',
    'avg', 'count', 'group_concat', 'sum',
    'acos', 'acosh', 'asin',
    'asinh', 'atan', 'atan2', 'atanh', 'ceil', 'cos', 'cosh', 'degrees', 'exp', 'floor', 'ln', 'log', 'log2', 'log10',
    'mod', 'pi', 'pow', 'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'trunc',
]


# https://www.sqlite.org/lang_corefunc.html

# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def abs(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def char(*x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def coalesce(*x) -> expr.SqlExpr:
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def hex(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def ifnull(x, y) -> expr.SqlExpr:
    """x or Y"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def iff(x, y, z) -> expr.SqlExpr:
    """x?y:z"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def instr(x, y) -> expr.SqlExpr:
    """x.find(y)"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def length(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def likelihood(x, y) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def lower(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def ltrim(x, y=None) -> expr.SqlExpr:
    pass


# noinspection PyShadowingBuiltins
@overload
def max(x) -> expr.SqlAggregateFunc:
    pass


# noinspection PyShadowingBuiltins
@overload
def max(x, *other) -> expr.SqlExpr:
    pass


# noinspection PyShadowingBuiltins
def max(x, *other):
    if len(other) == 0:
        return expr.SqlAggregateFunc('MAX', x)
    else:
        return expr.SqlFuncOper('MAX', x, *other)


# noinspection PyShadowingBuiltins
@overload
def min(x) -> expr.SqlAggregateFunc:
    pass


# noinspection PyShadowingBuiltins
@overload
def min(x, *other) -> expr.SqlExpr:
    pass


# noinspection PyShadowingBuiltins
def min(x, *other):
    if len(other) == 0:
        return expr.SqlAggregateFunc('MIN', x)
    else:
        return expr.SqlFuncOper('MIN', x, *other)


# noinspection PyUnusedLocal
@as_func_expr
def nullif(x, y) -> expr.SqlExpr:
    """x if x != y else None"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def quote(x) -> expr.SqlExpr:
    pass


@as_func_expr
def random() -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def randomblob(n: int) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def replace(x, y, z) -> expr.SqlExpr:
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def round(x, y=None) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def rtrim(x, y=None) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def sign(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def substr(x, y, z=None) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def trim(x, y=None) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def typeof(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def unhex(x, y=None) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def unicode(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def unlikely(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def upper(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def zeroblob(n: int) -> expr.SqlExpr:
    pass


# https://www.sqlite.org/lang_aggfunc.html#aggfunclist

# noinspection PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def avg(x) -> expr.SqlAggregateFunc:
    pass


def count(x=None) -> expr.SqlAggregateFunc:
    if x is None:
        return expr.SqlAggregateFunc('COUNT', expr.SqlLiteral('*'))
    else:
        return expr.SqlAggregateFunc('COUNT', x)


# noinspection PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def group_concat(x, y=None) -> expr.SqlAggregateFunc:
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def sum(x) -> expr.SqlAggregateFunc:
    pass


# noinspection PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def total(x) -> expr.SqlAggregateFunc:
    pass


# https://www.sqlite.org/lang_mathfunc.html

# noinspection PyUnusedLocal
@as_func_expr
def acos(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def acosh(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def asin(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def asinh(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def atan(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def atan2(y, x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def atanh(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def ceil(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def cos(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def cosh(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def degrees(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def exp(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def floor(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def ln(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def log2(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def log10(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def log(b, x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def mod(x, y) -> expr.SqlExpr:
    pass


@as_func_expr
def pi() -> expr.SqlExpr:
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def pow(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def radians(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def sin(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def sinh(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def sqrt(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def tan(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def tanh(x) -> expr.SqlExpr:
    pass


# noinspection PyUnusedLocal
@as_func_expr
def trunc(x) -> expr.SqlExpr:
    pass
