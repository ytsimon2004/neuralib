"""
SQL functions.

covers
* https://www.sqlite.org/lang_corefunc.html
* https://www.sqlite.org/lang_aggfunc.html
* https://www.sqlite.org/lang_mathfunc.html
"""
from __future__ import annotations

from typing import overload, TypeVar, Any

from typing_extensions import LiteralString

from . import expr
from .func_dec import as_func_expr

__all__ = [
    'abs', 'changes', 'char', 'coalesce', 'format',
    'hex', 'ifnull', 'iif', 'instr', 'last_insert_rowid', 'length', 'likelihood', 'likely', 'lower', 'ltrim',
    'max', 'min', 'nullif', 'octet_length', 'printf', 'quote', 'random', 'randomblob', 'replace', 'round', 'rtrim',
    'sign', 'sqlite_source_id', 'sqlite_version', 'substr', 'total_changes', 'trim', 'typeof',
    'unhex', 'unicode', 'unlikely', 'upper', 'zeroblob',
    'avg', 'count', 'group_concat', 'sum',
    'acos', 'acosh', 'asin',
    'asinh', 'atan', 'atan2', 'atanh', 'ceil', 'cos', 'cosh', 'degrees', 'exp', 'floor', 'ln', 'log', 'log2', 'log10',
    'mod', 'pi', 'pow', 'radians', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'trunc',
]

T = TypeVar('T')
N = TypeVar('N', int, float)

# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def abs(x: N) -> N:
    """https://www.sqlite.org/lang_corefunc.html#abs"""
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def changes() -> int:
    """https://www.sqlite.org/lang_corefunc.html#changes"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def char(*x) -> str:
    """https://www.sqlite.org/lang_corefunc.html#char"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def coalesce(*x: T) -> T | None:
    """https://www.sqlite.org/lang_corefunc.html#coalesce"""
    pass


# https://www.sqlite.org/lang_corefunc.html#concat by func_stat
# https://www.sqlite.org/lang_corefunc.html#glob by func_stat

# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def format(fmt: LiteralString, *x) -> str:
    """https://www.sqlite.org/lang_corefunc.html#format"""
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def hex(x: bytes) -> str:
    """https://www.sqlite.org/lang_corefunc.html#hex"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def ifnull(x: T, y: T) -> T:
    """
    ``x?:y``

    https://www.sqlite.org/lang_corefunc.html#ifnull

    """
    pass


# noinspection PyUnusedLocal
@as_func_expr
def iif(x, y: T, z: T) -> T:
    """
    ``x?y:z``

    https://www.sqlite.org/lang_corefunc.html#iif
    """
    pass


# noinspection PyUnusedLocal
@as_func_expr
def instr(x: str, y: str) -> int:
    """
    ``x.find(y)``

    https://www.sqlite.org/lang_corefunc.html#instr
    """
    pass


# noinspection PyUnusedLocal
@as_func_expr
def last_insert_rowid() -> int:
    """https://www.sqlite.org/lang_corefunc.html#last_insert_rowid"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def length(x: str | bytes) -> int:
    """https://www.sqlite.org/lang_corefunc.html#length"""
    pass


# https://www.sqlite.org/lang_corefunc.html#like by func_stat

# noinspection PyUnusedLocal
@as_func_expr
def likelihood(x: T, y: float) -> T:
    """https://www.sqlite.org/lang_corefunc.html#likelihood"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def likely(x: T) -> T:
    """https://www.sqlite.org/lang_corefunc.html#likely"""
    pass


# TODO https://www.sqlite.org/lang_corefunc.html#load_extension

# noinspection PyUnusedLocal
@as_func_expr
def lower(x: str) -> str:
    """https://www.sqlite.org/lang_corefunc.html#lower"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def ltrim(x: str, y: str = None) -> str:
    """https://www.sqlite.org/lang_corefunc.html#ltrim"""
    pass


# noinspection PyShadowingBuiltins
@overload
def max(x: N) -> N:
    pass


# noinspection PyShadowingBuiltins
@overload
def max(x: N, *other: N) -> N:
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
def min(x: N) -> N:
    pass


# noinspection PyShadowingBuiltins
@overload
def min(x: N, *other: N) -> N:
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
def nullif(x: T, y: T) -> T | None:
    """
    ``x if x != y else None``

    https://www.sqlite.org/lang_corefunc.html#nullif
    """
    pass


# noinspection PyUnusedLocal
@as_func_expr
def octet_length(x: str) -> int:
    """https://www.sqlite.org/lang_corefunc.html#octet_length"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def printf(fmt: LiteralString, *x) -> str:
    """https://www.sqlite.org/lang_corefunc.html#printf"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def quote(x: str) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#quote"""
    pass


@as_func_expr
def random() -> int:
    """https://www.sqlite.org/lang_corefunc.html#random"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def randomblob(n: int) -> bytes:
    """https://www.sqlite.org/lang_corefunc.html#randomblob"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def replace(x: str, y: str, z: str) -> str:
    """https://www.sqlite.org/lang_corefunc.html#replace"""
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def round(x: float, y: int = None) -> float:
    """https://www.sqlite.org/lang_corefunc.html#round"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def rtrim(x: str, y: str = None) -> str:
    """https://www.sqlite.org/lang_corefunc.html#rtrim"""
    pass


@overload
def sign(x: N) -> int:
    pass


@overload
def sign(x: Any) -> None:
    pass

# noinspection PyUnusedLocal
@as_func_expr
def sign(x):
    """https://www.sqlite.org/lang_corefunc.html#sign"""
    pass


# TODO https://www.sqlite.org/lang_corefunc.html#soundex
# https://www.sqlite.org/lang_corefunc.html#sqlite_compileoption_get by Connection.sqlite_compileoption_get
# https://www.sqlite.org/lang_corefunc.html#sqlite_compileoption_used by Connection.sqlite_compileoption_used
# TODO https://www.sqlite.org/lang_corefunc.html#sqlite_offset

# noinspection PyUnusedLocal
@as_func_expr
def sqlite_source_id() -> str:
    """https://www.sqlite.org/lang_corefunc.html#sqlite_source_id"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def sqlite_version() -> str:
    """https://www.sqlite.org/lang_corefunc.html#sqlite_version"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def substr(x: str, y: int, z: int = None) -> str:
    """https://www.sqlite.org/lang_corefunc.html#substr"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def total_changes() -> int:
    """https://www.sqlite.org/lang_corefunc.html#total_changes"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def trim(x: str, y: str = None) -> str:
    """https://www.sqlite.org/lang_corefunc.html#trim"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def typeof(x) -> str:
    """https://www.sqlite.org/lang_corefunc.html#typeof"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def unhex(x: str, y=None) -> bytes:
    """https://www.sqlite.org/lang_corefunc.html#unhex"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def unicode(x: str) -> int:
    """https://www.sqlite.org/lang_corefunc.html#unicode"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def unlikely(x: T) -> T:
    """https://www.sqlite.org/lang_corefunc.html#unlikely"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def upper(x: str) -> str:
    """https://www.sqlite.org/lang_corefunc.html#upper"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def zeroblob(n: int) -> bytes:
    """https://www.sqlite.org/lang_corefunc.html#zeroblob"""
    pass


# noinspection PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def avg(x) -> float:
    """https://www.sqlite.org/lang_aggfunc.html#avg"""
    pass


# noinspection PyUnusedLocal
def count(x=None) -> int:
    """https://www.sqlite.org/lang_aggfunc.html#count"""
    pass


# noinspection PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def group_concat(x, y=None) -> str:
    """https://www.sqlite.org/lang_aggfunc.html#group_concat"""
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def sum(x) -> float:
    """https://www.sqlite.org/lang_aggfunc.html#sumunc"""
    pass


# noinspection PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def total(x) -> int:
    """https://www.sqlite.org/lang_aggfunc.html#sumunc"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def acos(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#acos"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def acosh(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#acosh"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def asin(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#asin"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def asinh(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#asinh"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def atan(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#atan"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def atan2(y: float, x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#atan2"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def atanh(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#atanh"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def ceil(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#ceil"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def cos(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#cos"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def cosh(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#cosh"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def degrees(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#degrees"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def exp(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#exp"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def floor(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#floor"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def ln(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#ln"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def log2(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#log2"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def log10(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#log"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def log(b: float, x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#log"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def mod(x: float, y: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#mod"""
    pass


@as_func_expr
def pi() -> float:
    """https://www.sqlite.org/lang_mathfunc.html#pi"""
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def pow(x: float, y: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#pow"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def radians(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#radians"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def sin(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#sin"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def sinh(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#sinh"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def sqrt(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#sqrt"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def tan(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#tan"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def tanh(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#tanh"""
    pass


# noinspection PyUnusedLocal
@as_func_expr
def trunc(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#trunc"""
    pass

