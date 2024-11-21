from __future__ import annotations

from pathlib import Path
from typing import Literal, get_type_hints, overload, ClassVar

import h5py
import numpy as np

from neuralib.util.verbose import fprint

# if TYPE_CHECKING:
#     import polars as pl
#     import pandas as pd

__all__ = [
    'H5pyDataWrapper', 'attr', 'group', 'array'
]


OPEN = Literal['r', 'r+', 'w', 'x', 'a']


class H5pyDataWrapper:
    READ_ONLY: ClassVar[bool]

    def __init_subclass__(cls, read_only=False, **kwargs):
        cls.READ_ONLY = read_only

    def __init__(self, file: str | Path | h5py.File | h5py.Group,
                 mode: OPEN = 'r'):
        """

        :param file: file path
        """

        if isinstance(file, Path):
            file = str(file)

        if self.READ_ONLY:
            if mode != 'r':
                fprint('read-only wrapper. force read mode')

            mode = 'r'

        if isinstance(file, str):
            file = h5py.File(file, mode)

        self.__file = file

    @property
    def file(self) -> h5py.File | h5py.Group:
        return self.__file

    def __del__(self):
        if isinstance(file := self.__file, h5py.File):
            file.close()

    def __getitem__(self, item):
        return self.__file[item]


def attr(name: str = None):
    return H5pyDataWrapperAttr(name)


def group(name: str = None):
    return H5pyDataWrapperGroup(name)


@overload
def array(key: str = None,
          chunks: bool | int | tuple[int, ...] = None,
          maxshape: int | tuple[int, ...] = None,
          compression: int | str = None,
          compression_opts: int | tuple = None,
          scaleoffset: int = None,
          shuffle: bool = None,
          fillvalue: int | float | str = None,
          **kwargs) -> np.ndarray:
    pass


def array(key: str = None, **kwargs) -> np.ndarray:
    return H5pyDataWrapperArray(key, **kwargs)


# def table(key: str = None, **kwargs) -> pd.DataFrame | pl.Dataframe:
#     return H5pyDataWrapperTable(key, **kwargs)

class H5pyDataWrapperAttr:
    __slots__ = '__attr', '__type'

    def __init__(self, name: str = None):
        self.__attr = name
        self.__type = None

    def __set_name__(self, owner, name):
        if not issubclass(owner, H5pyDataWrapper):
            raise TypeError('owner type not H5pyDataWrapper')

        if self.__attr is None:
            self.__attr = name
        self.__type = get_type_hints(owner).get(name, None)

    def __get__(self, instance: H5pyDataWrapper, owner):
        if instance is None:
            return self
        else:
            file: h5py.File = instance.file

            try:
                ret = file.attrs[self.__attr]
            except KeyError as e:
                raise AttributeError(self.__attr) from e

            if self.__type is not None:
                ret = self.__type(ret)
            return ret

    def __set__(self, instance: H5pyDataWrapper, value):
        try:
            instance.file.attrs[self.__attr] = value
        except OSError as e:
            raise AttributeError(self.__attr) from e

    def __delete__(self, instance: H5pyDataWrapper):
        try:
            del instance.file.attrs[self.__attr]
        except OSError as e:
            raise AttributeError(self.__attr) from e


class H5pyDataWrapperGroup:
    __slots__ = '__group', '__type'

    def __init__(self, group: str = None):
        self.__group = group
        self.__type = None

    def __set_name__(self, owner, name):
        if not issubclass(owner, H5pyDataWrapper):
            raise TypeError('owner type not H5pyDataWrapper')

        if self.__group is None:
            self.__group = name

        self.__type = get_type_hints(owner).get(name, None)
        if self.__type is not None and not issubclass(self.__type, H5pyDataWrapper):
            raise TypeError(f'h5py_group {name} type not H5pyDataWrapper')
        if self.__type is None:
            self.__type = H5pyDataWrapper

    def __get__(self, instance: H5pyDataWrapper, owner):
        if instance is None:
            return self
        else:
            file: h5py.File = instance.file

            try:
                return self.__type(file[self.__group])
            except KeyError:
                if instance.READ_ONLY:
                    raise

            try:
                return self.__type(file.create_group(self.__group))
            except ValueError as e:
                raise AttributeError(self.__group) from e


class H5pyDataWrapperArray:
    __slots__ = '__key', '__kwargs'

    def __init__(self, key: str = None, **kwargs):
        self.__key = key
        self.__kwargs = kwargs

    def __set_name__(self, owner, name):
        if not issubclass(owner, H5pyDataWrapper):
            raise TypeError('owner type not H5pyDataWrapper')

        if self.__key is None:
            self.__key = name

    def __get__(self, instance: H5pyDataWrapper, owner):
        if instance is None:
            return self
        else:
            file: h5py.File = instance.file

            try:
                ret = file[self.__key]
            except KeyError:
                pass
            else:
                return H5pyDataWrapperLazyArray(ret)

            return None

    def __set__(self, instance: H5pyDataWrapper, value: np.ndarray):
        file: h5py.File = instance.file
        try:
            file[self.__key]
        except KeyError:
            pass
        else:
            del file[self.__key]

        file.create_dataset(self.__key, data=value, **self.__kwargs)

    def __delete__(self, instance: H5pyDataWrapper):
        file: h5py.File = instance.file
        del file[self.__key]


class H5pyDataWrapperLazyArray:
    def __init__(self, data: h5py.Dataset):
        self.__data = data

    @property
    def ndim(self):
        return self.__data.ndim

    @property
    def dtype(self):
        return self.__data.dtype

    @property
    def shape(self):
        return self.__data.shape

    def __array__(self, *args, **kwargs):
        return np.asarray(self.__data, *args, **kwargs)

    def __getitem__(self, item):
        return np.asarray(self.__data[item])

# class H5pyDataWrapperTable:
#     __slots__ = '__key', '__table', '__kwargs'
#
#     def __init__(self, key: str = None, **kwargs):
#         self.__key = key
#         self.__table = None
#         self.__kwargs = kwargs
#
#     def __set_name__(self, owner, name):
#         if not issubclass(owner, H5pyDataWrapper):
#             raise TypeError('owner type not H5pyDataWrapper')
#
#         if self.__key is None:
#             self.__key = name
#
#         table_type = get_type_hints(owner).get(name, None)
#
#         try:
#             import pandas as pd
#             if issubclass(table_type, pd.DataFrame):
#                 self.__table = pd.read_hdf
#         except ImportError:
#             pass
#
#         if self.__table is None:
#             try:
#                 import polars as pl
#                 if issubclass(table_type, pl.DataFrame):
#                     def polars_table(df, **kwargs):
#                         import pandas as pd
#                         return pl.from_pandas(pd.read_hdf(df, **kwargs))
#
#                     self.__table = polars_table
#             except ImportError:
#                 pass
#
#     def __get__(self, instance: H5pyDataWrapper, owner):
#         if instance is None:
#             return self
#         else:
#             file: h5py.File = instance.file
#
#             try:
#                 ret = file[self.__key]
#             except KeyError:
#                 pass
#             else:
#                 return self.__table(ret, **self.__kwargs)
#
#             return None
#
#     def __set__(self, instance: H5pyDataWrapper, value: np.ndarray):
#         raise RuntimeError('unsupported now')
#
#     def __delete__(self, instance: H5pyDataWrapper):
#         file: h5py.File = instance.file
#         del file[self.__key]
