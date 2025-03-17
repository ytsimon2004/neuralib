import re
from collections.abc import Collection
from dataclasses import is_dataclass
from pathlib import Path
from typing import TypeVar, overload, Literal, Any

from neuralib.typing import PathLike
from neuralib.typing.func import is_namedtuple
from neuralib.util.verbose import fprint

__all__ = ['uglob',
           'filter_matched',
           'joinn',
           'ensure_dir',
           'keys_with_value',
           'cls_hasattr']


def uglob(directory: PathLike,
          pattern: str,
          is_dir: bool = False) -> Path:
    """
    Use glob pattern to find the unique file in the directory.

    :param directory: Directory
    :param pattern: Glob pattern
    :param is_dir: Is the pattern point to a directory?
    :return: The unique path
    :raise FileNotFoundError: the unique path is not existed.
    :raise NotADirectoryError: *directory* is not a directory
    :raise RuntimeError: more than one path are found.
    """
    directory = Path(directory)

    if not directory.exists():
        raise FileNotFoundError(f'{directory} not exit')

    if not directory.is_dir():
        raise NotADirectoryError(f'{directory} is not a directory')

    f = list(directory.glob(pattern))

    if is_dir:
        f = [ff for ff in f if ff.is_dir()]
    else:
        f = [ff for ff in f if not ff.is_dir()]

    if len(f) == 0:
        t = 'directory' if is_dir else 'file'
        raise FileNotFoundError(f'{directory} not have {t} with the pattern: {pattern}')
    elif len(f) == 1:
        return f[0]
    else:
        f.sort()
        t = 'directories' if is_dir else 'files'
        raise RuntimeError(f'multiple {t} were found in {directory} with the pattern {pattern} >>> {f}')


def filter_matched(pattern: str, strings: list[str]) -> list[str]:
    """
    Filter a list of string element that match the pattern.

    :param pattern: Regular expression
    :param strings: List of strings to find the pattern
    :return:
    """
    return list(filter(re.compile(pattern).match, strings))


def ensure_dir(p: PathLike, verbose: bool = True) -> Path:
    """
    Ensure the path is a directory. Create if it is not exist.

    :param p: path to be checked
    :param verbose: print verbose message
    """
    p = Path(p)

    if not p.exists():
        p.mkdir(parents=True)

        if verbose:
            fprint(f'create dir {p}', vtype='io')

    if not p.is_dir():
        raise NotADirectoryError(f'not a dir: {p}')

    return p


def joinn(sep: str, *part: str | None) -> str:
    """join non-None str with sep."""
    return sep.join([str(it) for it in part if it is not None])


# ============================== #

KT = TypeVar('KT')
VT = TypeVar('VT')


@overload
def keys_with_value(dy: dict[KT, Any | Collection[VT]],
                    value: VT,
                    to_item: Literal[False] = False) -> list[KT]:
    pass


@overload
def keys_with_value(dy: dict[KT, VT | Collection[VT]],
                    value: VT,
                    to_item: Literal[True] = True) -> KT:
    pass


def keys_with_value(dy, value, to_item=False):
    """
    Get keys from a dict that are associated with the value.

    Supports value types: str, int, float (with tolerance), and any collection types.

    :param dy: The value to match against the dictionary values
    :param value: The value to match against the dictionary value
    :param to_item: List of item to item if unique
    :return: A list of keys whose values match the provided value
    """
    matching_keys = []

    def _float_eq(v1, v2, tol=1e-9) -> bool:
        return abs(v1 - v2) < tol

    for key, val in dy.items():
        if isinstance(val, float) and isinstance(value, float):
            if _float_eq(val, value):
                matching_keys.append(key)
        elif isinstance(val, (str, int, Collection)) or is_namedtuple(val) or is_dataclass(val):
            if val == value:
                matching_keys.append(key)

    if to_item:
        if len(matching_keys) == 0:
            raise KeyError(f'{value} does not match any keys')
        elif len(matching_keys) == 1:
            return matching_keys[0]
        else:
            raise ValueError(f'multiple match keys: {matching_keys}')
    else:
        return matching_keys


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
