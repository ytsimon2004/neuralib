from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import Callable, Literal, overload, TypeVar, Generic

import polars as pl
from typing_extensions import Self

__all__ = ['DataFrameWrapper']


class DataFrameWrapper(metaclass=abc.ABCMeta):
    """
    A polars dataframe wrapper that support common dataframe structure operations.
    It is used on a custom class that contains a dataframe as its main data, this class
    support the custom class dataframe operations and return itself.

    For example, for a custom class::

        class A(DataFrameWrapper):
            data : pl.DataFrame

            def dataframe(self, ...): ... # implement abc method

    we could do a filtering operation::

        a = A(pl.read_csv(...))
        a = a.filter(pl.col('A') == 'A')

    """

    @overload
    def dataframe(self) -> pl.DataFrame:
        pass

    @overload
    def dataframe(self, dataframe: pl.DataFrame, may_inplace=True) -> Self:
        pass

    @abc.abstractmethod
    def dataframe(self, dataframe: pl.DataFrame = None, may_inplace=True):
        pass

    def __len__(self) -> int:
        return len(self.dataframe())

    @property
    def columns(self) -> list[str]:
        return self.dataframe().columns

    @property
    def schema(self) -> pl.Schema:
        return self.dataframe().schema

    def __getitem__(self, item):
        return self.dataframe().__getitem__(item)

    def lazy(self) -> LazyDataFrameWrapper[Self]:
        return LazyDataFrameWrapper(self, self.dataframe().lazy())

    def rename(self, mapping: dict[str, str] | Callable[[str], str]) -> Self:
        return self.dataframe(self.dataframe().rename(mapping))

    def filter(self, *predicates, **constraints) -> Self:
        return self.dataframe(self.dataframe().filter(*predicates, **constraints))

    def sort(self,
             by, *more_by,
             descending: bool | Sequence[bool] = False,
             nulls_last: bool | Sequence[bool] = False,
             multithreaded: bool = True,
             maintain_order: bool = False) -> Self:
        return self.dataframe(self.dataframe().sort(by, *more_by,
                                                    descending=descending, nulls_last=nulls_last,
                                                    multithreaded=multithreaded, maintain_order=maintain_order))

    def drop(self, *columns, strict: bool = True) -> Self:
        return self.dataframe(self.dataframe().drop(*columns, strict=strict))

    @overload
    def partition_by(self, by, *more_by,
                     maintain_order: bool = True,
                     include_key: bool = True,
                     as_dict: Literal[False] = ...) -> list[Self]:
        pass

    @overload
    def partition_by(self, by, *more_by,
                     maintain_order: bool = ...,
                     include_key: bool = ...,
                     as_dict: Literal[True]) -> dict[tuple[object, ...], Self]:
        pass

    def partition_by(self, by, *more_by,
                     maintain_order: bool = True,
                     include_key: bool = True,
                     as_dict: bool = False):
        dataframe = self.dataframe().partition_by(by, *more_by,
                                                  maintain_order=maintain_order, include_key=include_key, as_dict=as_dict)
        if as_dict:
            return {k: self.dataframe(it, may_inplace=False) for k, it in dataframe.items()}
        else:
            return [self.dataframe(it, may_inplace=False) for it in dataframe]

    def with_columns(self, *exprs, **named_exprs) -> Self:
        return self.dataframe(self.dataframe().with_columns(*exprs, **named_exprs))

    def join(self, other: pl.DataFrame, on, how="inner", *,
             left_on=None,
             right_on=None,
             suffix: str = "_right",
             validate="m:m",
             join_nulls: bool = False,
             coalesce: bool | None = None) -> Self:
        return self.dataframe(self.dataframe().join(other, on, how,
                                                    left_on=left_on, right_on=right_on, suffix=suffix,
                                                    validate=validate, join_nulls=join_nulls, coalesce=coalesce))

    def group_by(self, *by,
                 maintain_order: bool = False,
                 **named_by) -> pl.GroupBy:
        return self.dataframe().group_by(*by, maintain_order=maintain_order, **named_by)


T = TypeVar('T', bound=DataFrameWrapper)


class LazyDataFrameWrapper(Generic[T]):
    def __init__(self, wrapper: T, lazy: pl.LazyFrame):
        self.__wrapper = wrapper
        self.__lazy = lazy

    def lazy(self) -> LazyDataFrameWrapper[T]:
        return self

    def collect(self, **kwargs) -> T:
        return self.__wrapper.dataframe(self.__lazy.collect(**kwargs))

    def rename(self, mapping: dict[str, str] | Callable[[str], str]) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.rename(mapping))

    def filter(self, *predicates, **constraints) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.filter(*predicates, **constraints))

    def sort(self,
             by, *more_by,
             descending: bool | Sequence[bool] = False,
             nulls_last: bool | Sequence[bool] = False,
             multithreaded: bool = True,
             maintain_order: bool = False) -> Self:
        df = self.__lazy.sort(by, *more_by, descending=descending, nulls_last=nulls_last,
                              multithreaded=multithreaded, maintain_order=maintain_order)
        return LazyDataFrameWrapper(self.__wrapper, df)

    def drop(self, *columns, strict: bool = True) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.drop(*columns, strict=strict))

    def with_columns(self, *exprs, **named_exprs) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.with_columns(*exprs, **named_exprs))

    def join(self, other: pl.DataFrame | pl.LazyFrame, on, how="inner", *,
             left_on=None,
             right_on=None,
             suffix: str = "_right",
             validate="m:m",
             join_nulls: bool = False,
             coalesce: bool | None = None) -> Self:
        if not isinstance(other, pl.LazyFrame):
            other = other.lazy()
        df = self.__lazy.join(other, on, how, left_on=left_on, right_on=right_on, suffix=suffix,
                              validate=validate, join_nulls=join_nulls, coalesce=coalesce)
        return LazyDataFrameWrapper(self.__wrapper, df)
