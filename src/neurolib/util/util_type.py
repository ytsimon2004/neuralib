from __future__ import annotations

from io import BufferedIOBase, BufferedReader
from pathlib import Path
from typing import BinaryIO, Union, TypeVar

import numpy as np
import pandas as pd
import polars as pl

__all__ = [
    'ArrayLike',
    'PathLike',
    #
    'Series',
    'DataFrame'
]

T = TypeVar('T')
ArrayLike = Union[np.ndarray, list[T], tuple[T, ...]]

PathLike = Union[str, Path, bytes, BinaryIO, BufferedIOBase, BufferedReader]

#
Series = Union[pd.Series, pl.Series]
DataFrame = Union[pd.DataFrame, pl.DataFrame]