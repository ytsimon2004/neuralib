from __future__ import annotations

import re
from pathlib import Path
from typing import TypeVar

from neuralib.typing import PathLike
from neuralib.util.verbose import fprint

__all__ = ['uglob',
           'glob_re',
           'joinn',
           'ensure_dir',
           'key_from_value',
           'cls_hasattr']


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

    if not p.exists():
        p.mkdir(parents=True)

        if verbose:
            fprint(f'create dir {p}', vtype='io')

    if not p.is_dir():
        raise NotADirectoryError(f'not a dir: {p}')


def joinn(sep: str, *part: str | None) -> str:
    """join not-None str"""
    return sep.join(str(it) for it in part if it is not None)


# ============================== #

KT = TypeVar('KT')
VT = TypeVar('VT')


def key_from_value(d: dict[KT, list[VT] | VT], value: VT) -> KT | list[KT]:
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


def cls_hasattr(cls: type, attr: str) -> bool:
    """
    Check if attributes in class

    :param cls: The class to check for the attribute.
    :param attr: The name of the attribute to look for within the class and its hierarchy.
    :return: True if the class or any of its parent classes has the specified attribute, False otherwise.
    """
    if attr in getattr(cls, '__annotations__', {}):
        return True

    for c in cls.mro()[1:]:  # Skip the first class as it's already checked
        if attr in getattr(c, '__annotations__', {}):
            return True

    return False
