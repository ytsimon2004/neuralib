"""
SQL window functions.

https://www.sqlite.org/windowfunctions.html
https://www.sqlite.org/windowfunctions.html#built_in_window_functions
"""

from . import expr
from .expr import SqlWindowFunc
from .func_dec import as_func_expr

__all__ = [
    'window_def', 'row_number', 'rank', 'dense_rank', 'percent_rank', 'cume_dist', 'ntile', 'lag', 'lead',
    'first_value', 'last_value', 'nth_value'
]


def window_def(name: str = None, *, order_by: list = None, partition_by: list = None) -> expr.SqlWindowDef:
    return expr.SqlWindowDef(name).over(order_by=order_by, partition_by=partition_by)


@as_func_expr(func=SqlWindowFunc)
def row_number() -> SqlWindowFunc:
    raise NotImplementedError()


@as_func_expr(func=SqlWindowFunc)
def rank() -> SqlWindowFunc:
    raise NotImplementedError()


@as_func_expr(func=SqlWindowFunc)
def dense_rank() -> SqlWindowFunc:
    raise NotImplementedError()


@as_func_expr(func=SqlWindowFunc)
def percent_rank() -> SqlWindowFunc:
    raise NotImplementedError()


@as_func_expr(func=SqlWindowFunc)
def cume_dist() -> SqlWindowFunc:
    raise NotImplementedError()


# noinspection PyUnusedLocal
@as_func_expr(func=SqlWindowFunc)
def ntile(n: int) -> SqlWindowFunc:
    raise NotImplementedError()


# noinspection PyUnusedLocal
@as_func_expr(func=SqlWindowFunc)
def lag(expr, offset: int = None, default=None) -> SqlWindowFunc:
    raise NotImplementedError()


# noinspection PyUnusedLocal
@as_func_expr(func=SqlWindowFunc)
def lead(expr, offset: int = None, default=None) -> SqlWindowFunc:
    raise NotImplementedError()


# noinspection PyUnusedLocal
@as_func_expr(func=SqlWindowFunc)
def first_value(expr) -> SqlWindowFunc:
    raise NotImplementedError()


# noinspection PyUnusedLocal
@as_func_expr(func=SqlWindowFunc)
def last_value(expr) -> SqlWindowFunc:
    raise NotImplementedError()


# noinspection PyUnusedLocal
@as_func_expr(func=SqlWindowFunc)
def nth_value(expr, n: int) -> SqlWindowFunc:
    raise NotImplementedError()
