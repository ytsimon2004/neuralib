from __future__ import annotations

from io import BufferedIOBase, BufferedReader
from pathlib import Path
from typing import BinaryIO, Union, TypeVar, Any

import numpy as np
import pandas as pd
import polars as pl
from matplotlib.axes import Axes
from numpy.typing import NDArray

__all__ = [
    'ArrayLike',
    'ArrayLikeStr',
    'NDArrayInt',
    'NDArrayFloat',
    'NDArrayBool',
    'AxesArrayLike',
    #
    'PathLike',
    #
    'Series',
    'DataFrame',
    #
    'is_iterable'
]

T = TypeVar('T')
ArrayLike = Union[np.ndarray, list[T], tuple[T, ...]]

ArrayLikeStr = Union[NDArray[np.str_], list[str], tuple[str, ...]]

NDArrayInt = NDArray[np.int_]
NDArrayFloat = NDArray[np.float_]
NDArrayBool = NDArray[np.bool_]

PathLike = Union[str, Path, bytes, BinaryIO, BufferedIOBase, BufferedReader]

#
Series = Union[pd.Series, pl.Series]
DataFrame = Union[pd.DataFrame, pl.DataFrame]

#
AxesArrayLike = Union[np.ndarray, list[Axes]]


# ============ #

def is_iterable(val: Any) -> bool:
    """check value is iterable"""
    from collections.abc import Iterable
    return isinstance(val, Iterable) and not isinstance(val, str)
