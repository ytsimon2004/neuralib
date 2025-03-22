from typing import Any

import numpy as np

from .alias import ArrayLike

__all__ = [
    'is_iterable',
    'is_namedtuple',
    'is_numeric_arraylike',
    'flatten_arraylike',
    'array2str'
]


def is_iterable(val: Any) -> bool:
    """Check if a value is iterable and not a string.

    :param val: The value to be checked.
    :return: True if the value is iterable and not a string, False otherwise
    """
    from collections.abc import Iterable
    return isinstance(val, Iterable) and not isinstance(val, str)


def is_numeric_arraylike(arr: ArrayLike) -> bool:
    """
    Check if is a numeric arraylike object

    :param arr: ArrayLike object
    :return:
    """
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)

    return np.issubdtype(arr.dtype, np.number)


# noinspection PyProtectedMember
def is_namedtuple(obj: Any) -> bool:
    """Check if is a namedtuple object"""
    if isinstance(obj, tuple) and hasattr(obj, '_fields'):
        if isinstance(obj._fields, tuple):
            return True

    return False


def flatten_arraylike(ls: list[ArrayLike]) -> list[Any]:
    """
    flatten the list

    :param ls:
    :return: flatten list
    """
    from itertools import chain
    return list(chain(*ls))


def array2str(x: ArrayLike, sep=' ') -> str:
    """Convert an array-like object to a string with a specified separator.

    :param x: 1D array-like object.
    :param sep: Separator used to join elements. Default is a space.
    :return: A string representation of the array with elements separated by `sep`.
    """
    return sep.join(map(str, x))
