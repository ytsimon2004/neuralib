from __future__ import annotations

import abc
from pathlib import Path
from typing import Literal, get_type_hints, overload, ClassVar, Generic, TypeVar, Any

import h5py
import numpy as np
import polars as pl
from numpy.core.numerictypes import issubdtype

from neuralib.util.verbose import fprint

__all__ = [
    'H5pyDataWrapper', 'attr', 'group', 'array', 'table'
]

T = TypeVar('T')
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


def table(key: str = None, backend: Literal['default', 'pytables'] = 'default', **kwargs) -> T:
    if backend == 'default':
        return H5pyDataWrapperTableDefault(key, **kwargs)
    elif backend == 'pytables':
        return H5pyDataWrapperTablePyTable(key, **kwargs)
    else:
        fprint(f'unknown util_h5py.table(backend={backend}). use default.', vtype='warning')
        return H5pyDataWrapperTableDefault(key, **kwargs)


class H5pyDataWrapperAttr:
    __slots__ = '__attr', '__type'

    def __init__(self, name: str = None):
        self.__attr = name
        self.__type = None

    def __set_name__(self, owner: type[H5pyDataWrapper], name: str):
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

    def __set_name__(self, owner: type[H5pyDataWrapper], name):
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

            return self.__type(file.require_group(self.__group))


class H5pyDataWrapperArray:
    __slots__ = '__key', '__kwargs'

    def __init__(self, key: str = None, **kwargs):
        self.__key = key
        self.__kwargs = kwargs

    def __set_name__(self, owner: type[H5pyDataWrapper], name):
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


class H5pyDataWrapperTable(Generic[T], metaclass=abc.ABCMeta):
    __slots__ = '__key', '_type', '_kwargs'

    def __init__(self, key: str = None, **kwargs):
        self.__key = key
        self._type = None
        self._kwargs = kwargs

    def __set_name__(self, owner, name):
        if not issubclass(owner, H5pyDataWrapper):
            raise TypeError('owner type not H5pyDataWrapper')

        if self.__key is None:
            self.__key = name

        self._type = get_type_hints(owner).get(name, None)

    @abc.abstractmethod
    def _get_table(self, table: h5py.Group) -> T:
        pass

    @abc.abstractmethod
    def _set_table(self, group: h5py.Group, table: T):
        pass

    def __get__(self, instance: H5pyDataWrapper, owner) -> T:
        if instance is None:
            return self
        else:
            file: h5py.File = instance.file

            try:
                ret = file[self.__key]
            except KeyError:
                pass
            else:
                return self._get_table(ret)

            return None

    def __set__(self, instance: H5pyDataWrapper, value: T):
        file: h5py.File = instance.file

        try:
            ret = file[self.__key]
            if not isinstance(ret, h5py.Group):
                del file[self.__key]
                raise KeyError
        except KeyError:
            ret = file.create_group(self.__key)

        self._set_table(ret, value)

    def __delete__(self, instance: H5pyDataWrapper):
        file: h5py.File = instance.file
        del file[self.__key]


class H5pyDataWrapperTableDefault(H5pyDataWrapperTable[pl.DataFrame]):

    def _get_table(self, table: h5py.Group) -> pl.DataFrame:
        import polars.datatypes as pty

        attrs = table['schema'].attrs
        content = table['table']

        schema = {
            name: getattr(pty, attrs[name])
            for name in attrs
        }

        data = {
            name: np.asarray(content[name])
            for name in schema
        }

        return pl.DataFrame(data=data, schema_overrides=schema)

    def _set_table(self, group: h5py.Group, table: pl.DataFrame):
        for name in table.schema:
            if (dtype := table.schema[name]).is_nested():
                raise RuntimeError(f'Do not support nested data type : {name} <{dtype}>')

        try:
            schema = group.create_group('schema')
        except ValueError:
            del group['schema']
            schema = group.create_group('schema')

        try:
            content = group.create_group('table')
        except ValueError:
            del group['table']
            content = group.create_group('table')

        for name in table.schema:
            schema.attrs[name] = str(table.schema[name])

        for name in table.columns:
            content.create_dataset(name, data=table[name].to_numpy())


class H5pyDataWrapperTablePyTable(H5pyDataWrapperTable[Any]):
    def __init__(self, key: str = None, **kwargs):
        super().__init__(key, **kwargs)

    def _get_table(self, table: h5py.Group):
        import pandas as pd
        df = pd.read_hdf(table, **self._kwargs)
        if issubdtype(self._type, pd.DataFrame):
            return df

        import polars as pl
        df = pl.from_pandas(df)
        if issubdtype(self._type, pl.DataFrame):
            return df

        try:
            from neuralib.util.util_polars import DataFrameWrapper
            if issubdtype(self._type, DataFrameWrapper):
                return self._type(df)
        except ImportError:
            # in ts-dev branch
            pass

        return df

    def _set_table(self, group: h5py.Group, table):
        raise RuntimeError('unsupported now')
