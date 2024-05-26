"""
SQL functions related to datetime.
"""
from .expr import SqlExpr
from .func_dec import as_func_expr

__all__ = [
    'date', 'time', 'datetime', 'julianday', 'unixepoch', 'strftime'
]


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def date(x) -> SqlExpr:
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def time(x) -> SqlExpr:
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def datetime(x) -> SqlExpr:
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def julianday(x) -> SqlExpr:
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def unixepoch(x) -> SqlExpr:
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def strftime(fmt, d) -> SqlExpr:
    pass
