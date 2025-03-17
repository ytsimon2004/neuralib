"""
SQL functions.

covers
* https://www.sqlite.org/lang_corefunc.html
* https://www.sqlite.org/lang_aggfunc.html
* https://www.sqlite.org/lang_mathfunc.html
"""
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


# noinspection PyShadowingBuiltins
@as_func_expr
def abs(x: N) -> N:
    """https://www.sqlite.org/lang_corefunc.html#abs"""
    import builtins
    return builtins.abs(x)


@as_func_expr
def changes() -> int:
    """https://www.sqlite.org/lang_corefunc.html#changes"""
    raise NotImplementedError()


# noinspection PyUnusedLocal
@as_func_expr
def char(*x) -> str:
    """https://www.sqlite.org/lang_corefunc.html#char"""
    raise NotImplementedError()


@as_func_expr
def coalesce(*x: T) -> T | None:
    """https://www.sqlite.org/lang_corefunc.html#coalesce"""
    for y in x:
        if y is not None:
            return y
    return None


# https://www.sqlite.org/lang_corefunc.html#concat by func_stat
# https://www.sqlite.org/lang_corefunc.html#glob by func_stat

# noinspection PyShadowingBuiltins
@as_func_expr
def format(fmt: LiteralString, *x) -> str:
    """https://www.sqlite.org/lang_corefunc.html#format"""
    return fmt % x


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def hex(x: bytes) -> str:
    """https://www.sqlite.org/lang_corefunc.html#hex"""
    raise NotImplementedError()


@as_func_expr
def ifnull(x: T, y: T) -> T:
    """
    ``x?:y``

    https://www.sqlite.org/lang_corefunc.html#ifnull

    """
    if x is not None:
        return x
    return y


@as_func_expr
def iif(x, y: T, z: T) -> T:
    """
    ``x?y:z``

    https://www.sqlite.org/lang_corefunc.html#iif
    """
    if x:
        return y
    else:
        return z


@as_func_expr
def instr(x: str, y: str) -> int:
    """
    ``x.find(y)``

    https://www.sqlite.org/lang_corefunc.html#instr
    """
    return x.find(y) + 1


@as_func_expr
def last_insert_rowid() -> int:
    """https://www.sqlite.org/lang_corefunc.html#last_insert_rowid"""
    raise NotImplementedError()


@as_func_expr
def length(x: str | bytes) -> int:
    """https://www.sqlite.org/lang_corefunc.html#length"""
    return len(x)


# https://www.sqlite.org/lang_corefunc.html#like by func_stat

@as_func_expr
def likelihood(x: T, y: float) -> T:
    """https://www.sqlite.org/lang_corefunc.html#likelihood"""
    return x


@as_func_expr
def likely(x: T) -> T:
    """https://www.sqlite.org/lang_corefunc.html#likely"""
    return x


# TODO https://www.sqlite.org/lang_corefunc.html#load_extension

@as_func_expr
def lower(x: str) -> str:
    """https://www.sqlite.org/lang_corefunc.html#lower"""
    return x.lower()


@as_func_expr
def ltrim(x: str, y: str = None) -> str:
    """https://www.sqlite.org/lang_corefunc.html#ltrim"""
    return x.lstrip(y)


# noinspection PyShadowingBuiltins
@overload
def max(x: N) -> N:
    raise NotImplementedError()


# noinspection PyShadowingBuiltins
@overload
def max(x: N, *other: N) -> N:
    raise NotImplementedError()


# noinspection PyShadowingBuiltins
def max(x, *other):
    """
    https://www.sqlite.org/lang_corefunc.html#max
    https://www.sqlite.org/lang_aggfunc.html#max_agg
    """
    if len(other) == 0:
        return expr.SqlAggregateFunc('MAX', x)
    else:
        return expr.SqlFuncOper('MAX', _max, x, *other)


def _max(*other):
    import builtins
    return builtins.max(*other)


# noinspection PyShadowingBuiltins
@overload
def min(x: N) -> N:
    raise NotImplementedError()


# noinspection PyShadowingBuiltins
@overload
def min(x: N, *other: N) -> N:
    raise NotImplementedError()


# noinspection PyShadowingBuiltins
def min(x, *other):
    """
    https://www.sqlite.org/lang_corefunc.html#min
    https://www.sqlite.org/lang_aggfunc.html#min_agg
    """
    if len(other) == 0:
        return expr.SqlAggregateFunc('MIN', x)
    else:
        return expr.SqlFuncOper('MIN', _min, x, *other)


def _min(*other):
    import builtins
    return builtins.min(*other)


@as_func_expr
def nullif(x: T, y: T) -> T | None:
    """
    ``x if x != y else None``

    https://www.sqlite.org/lang_corefunc.html#nullif
    """
    return x if x != y else None


# noinspection PyUnusedLocal
@as_func_expr
def octet_length(x: str) -> int:
    """https://www.sqlite.org/lang_corefunc.html#octet_length"""
    raise NotImplementedError()


@as_func_expr
def printf(fmt: LiteralString, *x) -> str:
    """https://www.sqlite.org/lang_corefunc.html#printf"""
    return fmt % x


# noinspection PyUnusedLocal
@as_func_expr
def quote(x: str) -> expr.SqlExpr:
    """https://www.sqlite.org/lang_corefunc.html#quote"""
    raise NotImplementedError()


@as_func_expr
def random() -> int:
    """https://www.sqlite.org/lang_corefunc.html#random"""
    raise NotImplementedError()


# noinspection PyUnusedLocal
@as_func_expr
def randomblob(n: int) -> bytes:
    """https://www.sqlite.org/lang_corefunc.html#randomblob"""
    raise NotImplementedError()


@as_func_expr
def replace(x: str, y: str, z: str) -> str:
    """https://www.sqlite.org/lang_corefunc.html#replace"""
    if len(y) == 0:
        return x
    else:
        return x.replace(y, z)


# noinspection PyShadowingBuiltins
@as_func_expr
def round(x: float, y: int = None) -> float:
    """https://www.sqlite.org/lang_corefunc.html#round"""
    import builtins
    if y is None:
        y = 0
    return builtins.round(x, y)


@as_func_expr
def rtrim(x: str, y: str = None) -> str:
    """https://www.sqlite.org/lang_corefunc.html#rtrim"""
    return x.rstrip(y)


@overload
def sign(x: N) -> int:
    pass


@overload
def sign(x: Any) -> None:
    pass


@as_func_expr
def sign(x):
    """https://www.sqlite.org/lang_corefunc.html#sign"""
    if isinstance(x, (int, float)):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0
    return None


# TODO https://www.sqlite.org/lang_corefunc.html#soundex
# https://www.sqlite.org/lang_corefunc.html#sqlite_compileoption_get by Connection.sqlite_compileoption_get
# https://www.sqlite.org/lang_corefunc.html#sqlite_compileoption_used by Connection.sqlite_compileoption_used
# TODO https://www.sqlite.org/lang_corefunc.html#sqlite_offset

@as_func_expr
def sqlite_source_id() -> str:
    """https://www.sqlite.org/lang_corefunc.html#sqlite_source_id"""
    raise NotImplementedError()


@as_func_expr
def sqlite_version() -> str:
    """https://www.sqlite.org/lang_corefunc.html#sqlite_version"""
    raise NotImplementedError()


@as_func_expr
def substr(x: str, y: int, z: int = None) -> str:
    """https://www.sqlite.org/lang_corefunc.html#substr"""
    if z is None:
        if y > 0:
            return x[y - 1:]
        else:
            return x[y:]
    elif z > 0:
        if y > 0:
            return x[y - 1:y + z - 1]
        else:
            return x[y:y + z]
    else:
        if y > 0:
            return x[y + z - 1:y - 1]
        else:
            return x[y + z:y]


@as_func_expr
def total_changes() -> int:
    """https://www.sqlite.org/lang_corefunc.html#total_changes"""
    raise NotImplementedError()


@as_func_expr
def trim(x: str, y: str = None) -> str:
    """https://www.sqlite.org/lang_corefunc.html#trim"""
    return x.strip(y)


# noinspection PyUnusedLocal
@as_func_expr
def typeof(x) -> str:
    """https://www.sqlite.org/lang_corefunc.html#typeof"""
    raise NotImplementedError()


# noinspection PyUnusedLocal
@as_func_expr
def unhex(x: str, y=None) -> bytes:
    """https://www.sqlite.org/lang_corefunc.html#unhex"""
    raise NotImplementedError()


# noinspection PyUnusedLocal
@as_func_expr
def unicode(x: str) -> int:
    """https://www.sqlite.org/lang_corefunc.html#unicode"""
    raise NotImplementedError()


@as_func_expr
def unlikely(x: T) -> T:
    """https://www.sqlite.org/lang_corefunc.html#unlikely"""
    return x


@as_func_expr
def upper(x: str) -> str:
    """https://www.sqlite.org/lang_corefunc.html#upper"""
    return x.upper()


# noinspection PyUnusedLocal
@as_func_expr
def zeroblob(n: int) -> bytes:
    """https://www.sqlite.org/lang_corefunc.html#zeroblob"""
    raise NotImplementedError()


# noinspection PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def avg(x) -> float:
    """https://www.sqlite.org/lang_aggfunc.html#avg"""
    raise NotImplementedError()


def count(x=None) -> int:
    """https://www.sqlite.org/lang_aggfunc.html#count"""
    if x is None:
        return expr.SqlAggregateFunc('COUNT', expr.SqlLiteral('*'))
    else:
        return expr.SqlAggregateFunc('COUNT', x)


# noinspection PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def group_concat(x, y=None) -> str:
    """https://www.sqlite.org/lang_aggfunc.html#group_concat"""
    raise NotImplementedError()


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def sum(x) -> float:
    """https://www.sqlite.org/lang_aggfunc.html#sumunc"""
    raise NotImplementedError()


# noinspection PyUnusedLocal
@as_func_expr(func=expr.SqlAggregateFunc)
def total(x) -> int:
    """https://www.sqlite.org/lang_aggfunc.html#sumunc"""
    raise NotImplementedError()


@as_func_expr
def acos(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#acos"""
    import math
    return math.acos(x)


@as_func_expr
def acosh(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#acosh"""
    import math
    return math.acosh(x)


@as_func_expr
def asin(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#asin"""
    import math
    return math.asin(x)


@as_func_expr
def asinh(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#asinh"""
    import math
    return math.asinh(x)


@as_func_expr
def atan(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#atan"""
    import math
    return math.atan(x)


@as_func_expr
def atan2(y: float, x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#atan2"""
    import math
    return math.atan2(y, x)


@as_func_expr
def atanh(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#atanh"""
    import math
    return math.atanh(x)


@as_func_expr
def ceil(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#ceil"""
    import math
    return math.ceil(x)


@as_func_expr
def cos(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#cos"""
    import math
    return math.cos(x)


@as_func_expr
def cosh(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#cosh"""
    import math
    return math.cosh(x)


@as_func_expr
def degrees(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#degrees"""
    import math
    return math.degrees(x)


@as_func_expr
def exp(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#exp"""
    import math
    return math.exp(x)


@as_func_expr
def floor(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#floor"""
    import math
    return math.floor(x)


@as_func_expr
def ln(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#ln"""
    import math
    return math.log(x)


@as_func_expr
def log2(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#log2"""
    import math
    return math.log2(x)


@as_func_expr
def log10(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#log"""
    import math
    return math.log10(x)


@as_func_expr
def log(b: float, x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#log"""
    import math
    return math.log(x, b)


@as_func_expr
def mod(x: float, y: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#mod"""
    return x % y


@as_func_expr
def pi() -> float:
    """https://www.sqlite.org/lang_mathfunc.html#pi"""
    import math
    return math.pi


@as_func_expr
def pow(x: float, y: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#pow"""
    import math
    return math.pow(x, y)


@as_func_expr
def radians(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#radians"""
    import math
    return math.radians(x)


@as_func_expr
def sin(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#sin"""
    import math
    return math.sin(x)


@as_func_expr
def sinh(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#sinh"""
    import math
    return math.sinh(x)


@as_func_expr
def sqrt(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#sqrt"""
    import math
    return math.sqrt(x)


@as_func_expr
def tan(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#tan"""
    import math
    return math.tan(x)


@as_func_expr
def tanh(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#tanh"""
    import math
    return math.tanh(x)


@as_func_expr
def trunc(x: float) -> float:
    """https://www.sqlite.org/lang_mathfunc.html#trunc"""
    import math
    return math.trunc(x)
