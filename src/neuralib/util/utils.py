from __future__ import annotations

import functools
import re
from pathlib import Path
from typing import TypeVar

from neuralib.typing import PathLike
from neuralib.util.util_verbose import fprint

__all__ = ['uglob',
           'glob_re',
           'joinn',
           'ensure_dir',
           'future_deprecate',
           'key_from_value']


def uglob(directory: PathLike,
          pattern: str,
          sort: bool = True,
          is_dir: bool = False) -> Path:
    """
    Unique glob the pattern in a directory

    :param directory: directory
    :param pattern: pattern string
    :param sort: if sort
    :param is_dir: only return if is a directory
    :return: unique path
    """
    if not isinstance(directory, Path):
        directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f'{directory} not exit')

    if not directory.is_dir():
        raise NotADirectoryError(f'{directory} is not a directory')

    f = list(directory.glob(pattern))

    if is_dir:
        f = [ff for ff in f if ff.is_dir()]

    if sort:
        f.sort()

    if len(f) == 0:
        raise FileNotFoundError(f'{directory} not have pattern: {pattern}')
    elif len(f) == 1:
        return f[0]
    else:
        raise RuntimeError(f'multiple files were found in {directory} in pattern {pattern} >>> {f}')


def glob_re(pattern: str, strings: list[str]) -> list[str]:
    """find list of str element fit for re pattern"""
    return list(filter(re.compile(pattern).match, strings))


def ensure_dir(p: PathLike, verbose: bool = True) -> None:
    """ensure the path, create if not exit

    :param p: path to be checked
    :param verbose: print verbose
    """
    p = Path(p)
    if not p.is_dir():
        raise NotADirectoryError(f'not a dir: {p}')

    if not p.exists():
        p.mkdir(parents=True)

        if verbose:
            fprint(f'create dir {p}', vtype='io')


def joinn(sep: str, *part: str | None) -> str:
    """join not-None str"""
    return sep.join(str(it) for it in part if it is not None)


def future_deprecate(f=None, *,
                     reason: str | None = None,
                     removal_version: str = None):
    """Mark deprecated functions.

    :param f: The function to be deprecated
    :param reason: The reason why the function is deprecated
    :param removal_version: Optional version or date when the function is planned to be removed
    :return:
    """
    if reason is None:
        reason = 'This function is deprecated and may be removed in future versions.'

    if removal_version is not None:
        reason += f' Scheduled for removal in version {removal_version}.'

    def _deprecated(f):
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


def key_from_value(d: dict[KT, VT], value: VT) -> KT | list[KT]:
    """Get dict key from dict value, supporting str, int, float, list, and tuple types for values."""
    matching_keys = []
    for key, val in d.items():
        if not isinstance(val, (str, int, float, list, tuple)):
            raise RuntimeError(f'value type: {type(val)} not support')
        else:
            if isinstance(val, (str, int, float)) and val == value:
                matching_keys.append(key)
            elif isinstance(val, (list, tuple)) and value in val:
                matching_keys.append(key)

    if not matching_keys:
        raise KeyError(f'Value {value} not found in the dictionary')
    else:
        return matching_keys[0] if len(matching_keys) == 1 else matching_keys
