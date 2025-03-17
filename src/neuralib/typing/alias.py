from io import BufferedIOBase, BufferedReader
from pathlib import Path
from typing import TypeVar, Union, BinaryIO

import numpy as np
import pandas as pd
import polars as pl
from matplotlib.axes import Axes
from numpy.typing import NDArray

__all__ = [
    'ArrayLike',
    'ArrayLikeStr',
    'AxesArray',
    #
    'PathLike',
    #
    'Series',
    'DataFrame'
]

# ===== #
# Array #
# ===== #

T = TypeVar('T')
"""Numeric"""

ArrayLike = Union[NDArray[T], list[T], tuple[T, ...], pd.Series, pl.Series]
"""Alias for array-like objects, including numpy arrays, lists, tuples, and series"""

ArrayLikeStr = Union[NDArray[np.str_], list[str], tuple[str, ...], pd.Series, pl.Series]
"""Alias for array-like objects of strings, including numpy arrays, lists, tuples, and series"""

AxesArray = Union[np.ndarray, list[Axes]]
"""Alias for matplotlib Axes numpy array"""

# ==== #
# Path #
# ==== #

PathLike = Union[str, Path, bytes, BinaryIO, BufferedIOBase, BufferedReader]
"""Alias for path-like objects, including strings, Paths, and file-like objects"""

# ================== #
# Series / DataFrame #
# ================== #

Series = Union[pd.Series, pl.Series]
"""Alias for series objects from pandas or polars"""

DataFrame = Union[pd.DataFrame, pl.DataFrame]
"""Alias for dataframe objects from pandas or polars"""
