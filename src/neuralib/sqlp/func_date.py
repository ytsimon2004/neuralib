"""
SQL functions related to datetime.

https://www.sqlite.org/lang_datefunc.html
"""
from .expr import SqlExpr
from .func_dec import as_func_expr

__all__ = [
    'date', 'time', 'datetime', 'julianday', 'unixepoch', 'strftime', 'timediff'
]


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def date(t, *m) -> SqlExpr:
    """

    https://www.sqlite.org/lang_datefunc.html

    :param t: time value
    :param m: modifier
    :return:
    """
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def time(t, *m) -> SqlExpr:
    """

    https://www.sqlite.org/lang_datefunc.html

    :param t: time value
    :param m: modifier
    :return:
    """
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def datetime(t, *m) -> SqlExpr:
    """

    https://www.sqlite.org/lang_datefunc.html

    :param t: time value
    :param m: modifier
    :return:
    """
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def julianday(t, *m) -> SqlExpr:
    """

    https://www.sqlite.org/lang_datefunc.html

    :param t: time value
    :param m: modifier
    :return:
    """
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def unixepoch(t, *m) -> SqlExpr:
    """

    https://www.sqlite.org/lang_datefunc.html

    :param t: time value
    :param m: modifier
    :return:
    """
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def strftime(fmt, t, *m) -> SqlExpr:
    """

    https://www.sqlite.org/lang_datefunc.html

    :param fmt: format
    :param t: time value
    :param m: modifier
    :return:
    """
    pass


# noinspection PyShadowingBuiltins,PyUnusedLocal
@as_func_expr
def timediff(a, b) -> SqlExpr:
    """

    https://www.sqlite.org/lang_datefunc.html

    :param a: time value
    :param b: time value
    :return:
    """
    pass
