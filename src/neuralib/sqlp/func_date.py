"""
SQL functions related to datetime.

https://www.sqlite.org/lang_datefunc.html
"""
from typing_extensions import LiteralString

from . import expr
from .func_dec import as_func_expr

__all__ = [
    'date', 'time', 'datetime', 'julianday', 'unixepoch', 'strftime', 'timediff'
]


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def date(t, *m) -> expr.SqlExpr:
    """

    https://www.sqlite.org/lang_datefunc.html

    :param t: time value
    :param m: modifier
    :return:
    """
    raise NotImplementedError()


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def time(t, *m) -> expr.SqlExpr:
    """

    https://www.sqlite.org/lang_datefunc.html

    :param t: time value
    :param m: modifier
    :return:
    """
    raise NotImplementedError()


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def datetime(t, *m) -> expr.SqlExpr:
    """

    https://www.sqlite.org/lang_datefunc.html

    :param t: time value
    :param m: modifier
    :return:
    """
    raise NotImplementedError()


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def julianday(t, *m) -> expr.SqlExpr:
    """

    https://www.sqlite.org/lang_datefunc.html

    :param t: time value
    :param m: modifier
    :return:
    """
    raise NotImplementedError()


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def unixepoch(t, *m) -> expr.SqlExpr:
    """

    https://www.sqlite.org/lang_datefunc.html

    :param t: time value
    :param m: modifier
    :return:
    """
    raise NotImplementedError()


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def strftime(fmt: LiteralString, t, *m) -> expr.SqlExpr:
    """

    https://www.sqlite.org/lang_datefunc.html

    :param fmt: format
    :param t: time value
    :param m: modifier
    :return:
    """
    raise NotImplementedError()


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def timediff(a, b) -> expr.SqlExpr:
    """

    https://www.sqlite.org/lang_datefunc.html

    :param a: time value
    :param b: time value
    :return:
    """
    raise NotImplementedError()
