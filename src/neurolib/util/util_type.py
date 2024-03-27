from __future__ import annotations

from io import BufferedIOBase, BufferedReader
from pathlib import Path
from typing import BinaryIO, Union, TypeVar, Any

import numpy as np
import pandas as pd
import polars as pl

__all__ = [
    'ArrayLike',
    'PathLike',
    #
    'Series',
    'DataFrame',
    #
    'is_iterable'
]

T = TypeVar('T')
ArrayLike = Union[np.ndarray, list[T], tuple[T, ...]]

PathLike = Union[str, Path, bytes, BinaryIO, BufferedIOBase, BufferedReader]

#
Series = Union[pd.Series, pl.Series]
DataFrame = Union[pd.DataFrame, pl.DataFrame]



def is_iterable(val: Any) -> bool:
    from collections.abc import Iterable
    return isinstance(val, Iterable) and not isinstance(val, str)
