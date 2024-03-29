from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import Any, TypeVar

from neuralib.util.util_type import PathLike, ArrayLike
from neuralib.util.util_verbose import fprint

__all__ = ['uglob',
           'glob_re',
           'joinn',
           'flat_ls',
           'array2str',
           'deprecated',
           'key_from_value']


def uglob(d: PathLike,
          pattern: str,
          sort: bool = True,
          is_dir: bool = False) -> Path:
    """Unique glob"""
    if not isinstance(d, Path):
        d = Path(d)

    if not d.is_dir():
        raise ValueError(f'{d} is not a directory')

    f = list(d.glob(pattern))

    if is_dir:
        f = [ff for ff in f if ff.is_dir()]

    if sort:
        f.sort()

    if len(f) == 0:
        raise FileNotFoundError(f'{d} not have pattern: {pattern}')
    elif len(f) == 1:
        return f[0]
    else:
        raise RuntimeError(f'multiple files were found in {d} in pattern {pattern} >>> {f}')


def glob_re(pattern: str, strings: list[str]) -> list[str]:
    """find list of str element fit for re pattern"""
    return list(filter(re.compile(pattern).match, strings))


def joinn(sep: str, *part: str | None) -> str:
    """join not-None str"""
    return sep.join(str(it) for it in part if it is not None)


def flat_ls(ls: list[ArrayLike]) -> list[Any]:
    """flatten the list"""
    from itertools import chain
    return list(chain(*ls))


def array2str(x: ArrayLike, sep=' ') -> str:
    """
    covert array to str with sep.
    i.e., used in csv writing of array in a specific cell

    :param x: 1D
    :param sep:
    :return:
    """
    return sep.join(map(str, x))


def deprecated(f=None, *, reason: str = None):
    """Mark deprecated functions.

    :param f:
    :param reason:
    :return:
    """
    if reason is None:
        reason = '...'

    def _deprecated(f):
        # TODO caller sensitive?
        warned = False

        @functools.wraps(f)
        def _deprecated_func(*args, **kwargs):
            nonlocal warned
            if not warned:
                fprint(f'use Deprecated function {f.__name__} : {reason}', vtype='warning')
                warned = True

            return f(*args, **kwargs)

        if f.__doc__ is None:
            _deprecated_func.__doc__ = "DEPRECATED."
        else:
            _deprecated_func.__doc__ = "DEPRECATED. " + f.__doc__

        return _deprecated_func

    if f is None:
        return _deprecated
    else:
        return _deprecated(f)


# ============================== #

KT = TypeVar('KT')
VT = TypeVar('VT')


def key_from_value(it: dict[KT, VT], value: VT) -> KT:
    """Get dict key from dict value"""
    kk = []
    for k, v in it.items():
        if (isinstance(v, (str, int)) and v == value) or (isinstance(v, (list, tuple)) and value in v):
            kk.append(k)

    if len(kk) == 0:
        raise KeyError(f'{value} not in the dict')
    if len(kk) == 1:
        return kk[0]
    else:
        raise RuntimeError(f'multiple key found by the value: {kk}: {value}')
