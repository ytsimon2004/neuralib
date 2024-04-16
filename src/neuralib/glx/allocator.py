from pathlib import Path
from typing import Protocol, Union

import numpy as np

__all__ = [
    'Allocator',
    'default_allocator',
    'inplace_allocator',
    'memmap_allocator',
]


class Allocator(Protocol):
    def __call__(self, shape: Union[tuple[int, ...], np.ndarray], dtype: np.dtype = None) -> np.ndarray:
        pass


def default_allocator() -> Allocator:
    def allocator(shape: Union[tuple[int, ...], np.ndarray], dtype: np.dtype = None):
        if isinstance(shape, np.ndarray):
            return np.empty_like(shape, dtype)
        return np.empty(shape, dtype)

    return allocator


def inplace_allocator() -> Allocator:
    def allocator(shape: Union[tuple[int, ...], np.ndarray], dtype: np.dtype = None):
        if isinstance(shape, np.ndarray):
            if dtype is not None:
                return shape.astype(dtype)
            return shape
        return np.empty(shape, dtype)

    return allocator


def memmap_allocator(file: Union[str, Path], **kwargs) -> Allocator:
    def allocator(shape: Union[tuple[int, ...], np.ndarray], dtype: np.dtype = None):
        if isinstance(shape, np.ndarray):
            if dtype is None:
                dtype = shape.dtype
            shape = shape.shape
        if dtype is None:
            dtype = np.float64
        return np.memmap(file, shape=shape, dtype=dtype, mode='w+', **kwargs)

    return allocator
