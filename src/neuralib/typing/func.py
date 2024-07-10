from __future__ import annotations

from typing import Any

from .alias import ArrayLike

__all__ = [
    'is_iterable',
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
