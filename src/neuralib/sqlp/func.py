"""
SQL functions.

covers
* https://www.sqlite.org/lang_corefunc.html
* https://www.sqlite.org/lang_aggfunc.html
* https://www.sqlite.org/lang_mathfunc.html
"""
from __future__ import annotations

from typing import overload

from typing_extensions import LiteralString

from . import expr
from .func_dec import as_func_expr

__all__ = [
    'abs', 'changes', 'char', 'coalesce', 'format',
    'hex', 'ifnull', 'iff', 'instr', 'last_insert_rowid', 'length', 'likelihood', 'likely', 'lower', 'ltrim',
    'max', 'min', 'nullif', 'octet_length', 'printf', 'quote', 'random', 'randomblob', 'replace', 'round', 'rtrim',
    'sign', 'sqlite_source_id', 'sqlite_version', 'substr', 'total_changes', 'trim', 'typeof',
    'unhex', 'unicode', 'unlikely', 'upper', 'zeroblob',
    'avg', 'count', 'group_concat', 'sum',
    'acos', 'acosh', 'asin',
    'asinh', 'atan', 'atan2', 'atanh', 'ceil', 'cos', 'cosh', 'degrees', 'exp', 'floor', 'ln', 'log', 'log2', 'log10',
    'mod', 'pi', 'pow', 'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'trunc',
]


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def abs(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#abs"""
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def changes(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#changes"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def char(*x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#char"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def coalesce(*x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#coalesce"""
    pass


# https://www.sqlite.org/lang_corefunc.html#concat by func_stat
# https://www.sqlite.org/lang_corefunc.html#glob by func_stat

# noinspection PyShadowingBuiltins,PyUnusedLocal
def format(fmt: LiteralString, *x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#format"""
    return expr.SqlFuncOper('FORMAT', expr.SqlLiteral(repr(fmt)), *x)


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def hex(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#hex"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def ifnull(x, y) -> expr.SqlExpr:
    """
    ``x?:y``

    https://www.sqlite.org/lang_corefunc.html#ifnull

    """
    pass


# noinspection PyUnusedLocal
@as_func_expr
def iff(x, y, z) -> expr.SqlExpr:
    """
    ``x?y:z``

    https://www.sqlite.org/lang_corefunc.html#iif
    """
    pass


# noinspection PyUnusedLocal
@as_func_expr
def instr(x, y) -> expr.SqlExpr:
    """
    ``x.find(y)``

    https://www.sqlite.org/lang_corefunc.html#instr
    """
    pass


# noinspection PyUnusedLocal
@as_func_expr
def last_insert_rowid() -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#last_insert_rowid"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def length(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#length"""
    pass


# https://www.sqlite.org/lang_corefunc.html#like by func_stat

# noinspection PyUnusedLocal
@as_func_expr
def likelihood(x, y) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#likelihood"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def likely(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#likely"""
    pass


# TODO https://www.sqlite.org/lang_corefunc.html#load_extension

# noinspection PyUnusedLocal
@as_func_expr
def lower(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#lower"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def ltrim(x, y=None) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#ltrim"""
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
    """
    https://www.sqlite.org/lang_corefunc.html#max
    https://www.sqlite.org/lang_aggfunc.html#max_agg
    """
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
    """
    https://www.sqlite.org/lang_corefunc.html#min
    https://www.sqlite.org/lang_aggfunc.html#min_agg
    """
    if len(other) == 0:
        return expr.SqlAggregateFunc('MIN', x)
    else:
        return expr.SqlFuncOper('MIN', x, *other)


# noinspection PyUnusedLocal
@as_func_expr
def nullif(x, y) -> expr.SqlExpr:
    """
    ``x if x != y else None``

    https://www.sqlite.org/lang_corefunc.html#nullif
    """
    pass


# noinspection PyUnusedLocal
@as_func_expr
def octet_length(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#octet_length"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def printf(fmt: LiteralString, *x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#printf"""
    return expr.SqlFuncOper('PRINTF', expr.SqlLiteral(repr(fmt)), *x)


# noinspection PyUnusedLocal
@as_func_expr
def quote(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#quote"""
    pass


@as_func_expr
def random() -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#random"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def randomblob(n: int) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#randomblob"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def replace(x, y, z) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#replace"""
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def round(x, y=None) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#round"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def rtrim(x, y=None) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#rtrim"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def sign(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#sign"""
    pass


# TODO https://www.sqlite.org/lang_corefunc.html#soundex
# TODO https://www.sqlite.org/lang_corefunc.html#sqlite_compileoption_get
# TODO https://www.sqlite.org/lang_corefunc.html#sqlite_compileoption_used
# TODO https://www.sqlite.org/lang_corefunc.html#sqlite_offset

# noinspection PyUnusedLocal
@as_func_expr
def sqlite_source_id() -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#sqlite_source_id"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def sqlite_version() -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#sqlite_version"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def substr(x, y, z=None) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#substr"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def total_changes() -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#total_changes"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def trim(x, y=None) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#trim"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def typeof(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#typeof"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def unhex(x, y=None) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#unhex"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def unicode(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#unicode"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def unlikely(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#unlikely"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def upper(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#upper"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def zeroblob(n: int) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#zeroblob"""
    pass


# noinspection PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def avg(x) -> expr.SqlAggregateFunc:
    """https://www.sqlite.org/lang_aggfunc.html#avg"""
    pass


def count(x=None) -> expr.SqlAggregateFunc:
    """https://www.sqlite.org/lang_aggfunc.html#count"""
    if x is None:
        return expr.SqlAggregateFunc('COUNT', expr.SqlLiteral('*'))
    else:
        return expr.SqlAggregateFunc('COUNT', x)


# noinspection PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def group_concat(x, y=None) -> expr.SqlAggregateFunc:
    """https://www.sqlite.org/lang_aggfunc.html#group_concat"""
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def sum(x) -> expr.SqlAggregateFunc:
    """https://www.sqlite.org/lang_aggfunc.html#sumunc"""
    pass


# noinspection PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def total(x) -> expr.SqlAggregateFunc:
    """https://www.sqlite.org/lang_aggfunc.html#sumunc"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def acos(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#acos"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def acosh(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#acosh"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def asin(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#asin"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def asinh(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#asinh"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def atan(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#atan"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def atan2(y, x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#atan2"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def atanh(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#atanh"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def ceil(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#ceil"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def cos(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#cos"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def cosh(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#cosh"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def degrees(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#degrees"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def exp(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#exp"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def floor(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#floor"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def ln(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#ln"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def log2(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#log2"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def log10(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#log"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def log(b, x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#log"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def mod(x, y) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#mod"""
    pass


@as_func_expr
def pi() -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#pi"""
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def pow(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#pow"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def radians(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#radians"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def sin(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#sin"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def sinh(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#sinh"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def sqrt(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#sqrt"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def tan(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#tan"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def tanh(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#tanh"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def trunc(x) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_mathfunc.html#trunc"""
    pass

