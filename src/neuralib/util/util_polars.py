from __future__ import annotations

import abc
from typing import Callable, Literal, overload, TypeVar, Generic, Any, TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence, Iterable, Collection
    from typing_extensions import Self, ParamSpec, Concatenate
    from polars import _typing as pty

    P = ParamSpec('P')

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

    def __array__(self, *args, **kwargs) -> np.ndarray:
        return self.dataframe().__array__(*args, **kwargs)

    def __dataframe__(self, *args, **kwargs):
        return self.dataframe().__dataframe__(*args, **kwargs)

    @overload
    def __getitem__(self, key: (
            str
            | tuple[[pty.MultiIndexSelector, pty.SingleColSelector]]
    )) -> pl.Series:
        pass

    @overload
    def __getitem__(self, key: (
            pty.SingleIndexSelector
            | pty.MultiIndexSelector
            | pty.MultiColSelector
            | tuple[pty.SingleIndexSelector, pty.MultiColSelector]
            | tuple[pty.MultiIndexSelector, pty.MultiColSelector]
    )) -> pl.DataFrame:
        pass

    def __getitem__(self, item):
        return self.dataframe().__getitem__(item)

    def lazy(self) -> LazyDataFrameWrapper[Self]:
        return LazyDataFrameWrapper(self, self.dataframe().lazy())

    def rename(self, mapping: dict[str, str] | Callable[[str], str]) -> Self:
        return self.dataframe(self.dataframe().rename(mapping))

    def filter(self, *predicates: pty.IntoExprColumn | Iterable[pty.IntoExprColumn] | bool | list[bool] | np.ndarray,
               **constraints: Any) -> Self:
        return self.dataframe(self.dataframe().filter(*predicates, **constraints))

    def slice(self, offset: int, length: int | None = None) -> Self:
        return self.dataframe(self.dataframe().slice(offset, length))

    def head(self, n: int = 5) -> Self:
        return self.dataframe(self.dataframe().head(n))

    def tail(self, n: int = 5) -> Self:
        return self.dataframe(self.dataframe().tail(n))

    def limit(self, n: int = 5) -> Self:
        return self.dataframe(self.dataframe().limit(n))

    @overload
    def sort(self,
             by: pty.IntoExpr | Iterable[[pty.IntoExpr]],
             *more_by: pty.IntoExpr,
             descending: bool | Sequence[bool] = False,
             nulls_last: bool | Sequence[bool] = False,
             multithreaded: bool = True,
             maintain_order: bool = False) -> Self:
        pass

    def sort(self, by, *more_by, **kwargs) -> Self:
        return self.dataframe(self.dataframe().sort(by, *more_by, **kwargs))

    def drop(self, *columns: pty.ColumnNameOrSelector | Iterable[pty.ColumnNameOrSelector],
             strict: bool = True) -> Self:
        return self.dataframe(self.dataframe().drop(*columns, strict=strict))

    def drop_nulls(self, subset: pty.ColumnNameOrSelector | Collection[pty.ColumnNameOrSelector]) -> Self:
        return self.dataframe(self.dataframe().drop_nulls(subset))

    def fill_null(self, value: Any | pl.Expr | None = None,
                  strategy: pty.FillNullStrategy | None = None,
                  limit: int | None = None, **kwargs) -> Self:
        return self.dataframe(self.dataframe().fill_null(value, strategy, limit, **kwargs))

    def fill_nan(self, value: pl.Expr | int | float | None = None) -> Self:
        return self.dataframe(self.dataframe().fill_nan(value))

    def clear(self, n: int = 5) -> Self:
        return self.dataframe(self.dataframe().clear(n))

    def clone(self) -> Self:
        return self.dataframe(self.dataframe(), may_inplace=False)

    @overload
    def partition_by(self, by: pty.ColumnNameOrSelector | Iterable[pty.ColumnNameOrSelector],
                     *more_by: pty.ColumnNameOrSelector,
                     maintain_order: bool = True,
                     include_key: bool = True,
                     as_dict: Literal[False] = ...) -> list[Self]:
        pass

    @overload
    def partition_by(self, by: pty.ColumnNameOrSelector | Iterable[pty.ColumnNameOrSelector],
                     *more_by: pty.ColumnNameOrSelector,
                     maintain_order: bool = ...,
                     include_key: bool = ...,
                     as_dict: Literal[True]) -> dict[tuple[object, ...], Self]:
        pass

    def partition_by(self, by, *more_by, as_dict=False, **kwargs):
        dataframe = self.dataframe().partition_by(by, *more_by, as_dict=as_dict, **kwargs)
        if as_dict:
            return {k: self.dataframe(it, may_inplace=False) for k, it in dataframe.items()}
        else:
            return [self.dataframe(it, may_inplace=False) for it in dataframe]

    def select(self, *exprs: pty.IntoExpr | Iterable[pty.IntoExpr],
               **named_exprs: pty.IntoExpr) -> Self:
        return self.dataframe(self.dataframe().select(*exprs, **named_exprs))

    def with_columns(self, *exprs: pty.IntoExpr | Iterable[pty.IntoExpr],
                     **named_exprs: pty.IntoExpr) -> Self:
        return self.dataframe(self.dataframe().with_columns(*exprs, **named_exprs))

    def with_row_index(self, name: str = "index", offset: int = 0) -> Self:
        return self.dataframe(self.dataframe().with_row_index(name, offset))

    # def shift(self, n: int = 1, *, fill_value: pty.IntoExpr | None = None) -> Self:
    #     return self.dataframe(self.dataframe().shift(n, fill_value=fill_value))

    @overload
    def join(self, other: pl.DataFrame | DataFrameWrapper,
             on: str | pl.Expr | Sequence[str | pl.Expr],
             how: pty.JoinStrategy = "inner", *,
             left_on=None,
             right_on=None,
             suffix: str = "_right",
             validate: pty.JoinValidation = "m:m",
             join_nulls: bool = False,
             coalesce: bool | None = None) -> Self:
        pass

    def join(self, other: pl.DataFrame | DataFrameWrapper, on, *args, **kwargs) -> Self:
        if isinstance(other, DataFrameWrapper):
            other = other.dataframe()

        return self.dataframe(self.dataframe().join(other, on, *args, **kwargs))

    def pipe(self, function: Callable[Concatenate[pl.DataFrame, P], pl.DataFrame],
             *args: P.args,
             **kwargs: P.kwargs) -> Self:
        return self.dataframe(self.dataframe().pipe(function, *args, **kwargs))

    def group_by(self, *by: pty.IntoExpr | Iterable[pty.IntoExpr],
                 maintain_order: bool = False,
                 **named_by: pty.IntoExpr) -> pl.GroupBy:
        return self.dataframe().group_by(*by, maintain_order=maintain_order, **named_by)


T = TypeVar('T', bound=DataFrameWrapper)


class LazyDataFrameWrapper(Generic[T]):
    __slots__ = '__wrapper', '__lazy'

    def __init__(self, wrapper: T, lazy: pl.LazyFrame):
        self.__wrapper = wrapper
        self.__lazy = lazy

    @property
    def columns(self) -> list[str]:
        return self.__lazy.columns

    @property
    def schema(self) -> pl.Schema:
        return self.__lazy.schema

    def lazy(self) -> LazyDataFrameWrapper[T]:
        return self

    def collect(self, **kwargs) -> T:
        return self.__wrapper.dataframe(self.__lazy.collect(**kwargs))

    def rename(self, mapping: dict[str, str] | Callable[[str], str]) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.rename(mapping))

    def slice(self, offset: int, length: int | None = None) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.slice(offset, length))

    def head(self, n: int = 5) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.head(n))

    def tail(self, n: int = 5) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.tail(n))

    def limit(self, n: int = 5) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.limit(n))

    def clear(self, n: int = 0) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.clear(n))

    def filter(self, *predicates: pty.IntoExprColumn | Iterable[pty.IntoExprColumn] | bool | list[bool] | np.ndarray,
               **constraints: Any) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.filter(*predicates, **constraints))

    @overload
    def sort(self,
             by: pty.IntoExpr | Iterable[[pty.IntoExpr]],
             *more_by: pty.IntoExpr,
             descending: bool | Sequence[bool] = False,
             nulls_last: bool | Sequence[bool] = False,
             multithreaded: bool = True,
             maintain_order: bool = False) -> Self:
        pass

    def sort(self, by, *more_by, **kwargs) -> Self:
        df = self.__lazy.sort(by, *more_by, **kwargs)
        return LazyDataFrameWrapper(self.__wrapper, df)

    def drop(self, *columns: pty.ColumnNameOrSelector | Iterable[pty.ColumnNameOrSelector],
             strict: bool = True) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.drop(*columns, strict=strict))

    def drop_nulls(self, subset: pty.ColumnNameOrSelector | Collection[pty.ColumnNameOrSelector]) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.drop_nulls(subset))

    def fill_null(self, value: Any | pl.Expr | None = None,
                  strategy: pty.FillNullStrategy | None = None,
                  limit: int | None = None, **kwargs) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.fill_null(value, strategy, limit, **kwargs))

    def fill_nan(self, value: pl.Expr | int | float | None = None) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.fill_nan(value))

    def select(self, *exprs: pty.IntoExpr | Iterable[pty.IntoExpr],
               **named_exprs: pty.IntoExpr) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.select(*exprs, **named_exprs))

    def with_columns(self, *exprs: pty.IntoExpr | Iterable[pty.IntoExpr],
                     **named_exprs: pty.IntoExpr) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.with_columns(*exprs, **named_exprs))

    def with_row_index(self, name: str = "index", offset: int = 0) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.with_row_index(name, offset))

    @overload
    def join(self, other: pl.DataFrame | DataFrameWrapper,
             on: str | pl.Expr | Sequence[str | pl.Expr],
             how: pty.JoinStrategy = "inner", *,
             left_on=None,
             right_on=None,
             suffix: str = "_right",
             validate: pty.JoinValidation = "m:m",
             join_nulls: bool = False,
             coalesce: bool | None = None) -> Self:
        pass

    def join(self, other: pl.DataFrame | DataFrameWrapper, on, *args, **kwargs) -> Self:
        if isinstance(other, DataFrameWrapper):
            other = other.dataframe()
        if not isinstance(other, pl.LazyFrame):
            other = other.lazy()

        df = self.__lazy.join(other, on, *args, **kwargs)
        return LazyDataFrameWrapper(self.__wrapper, df)

    def pipe(self, function: Callable[Concatenate[pl.LazyFrame, P], pl.LazyFrame],
             *args: P.args,
             **kwargs: P.kwargs) -> Self:
        return LazyDataFrameWrapper(self.__wrapper, self.__lazy.pipe(function, *args, **kwargs))


def helper_with_index_column(df: T,
                             column: str,
                             index: int | list[int] | np.ndarray | T,
                             maintain_order: bool = False,
                             strict: bool = False) -> T:
    """
    A help function to do the filter on an index column.

    :param df:
    :param column: index column
    :param index: index array
    :param maintain_order: keep the ordering of *index* in the returned dataframe.
    :param strict: all index in *index* should present in the returned dataframe. Otherwise, an error will be raised.
    :return:
    :raise RuntimeError: strict mode fail.
    """
    if isinstance(index, (int, np.integer)):
        ret = df.filter(pl.col(column) == index)
        if len(ret) == 0 and strict:
            raise RuntimeError(f'missing {column}: [{index}]')
    elif isinstance(index, type(df)):
        index = index[column].to_numpy()
    else:
        index = np.asarray(index)

    if strict:
        if len(miss := np.setdiff1d(index, df.dataframe()[column].unique().to_numpy())) > 0:
            raise RuntimeError(f'missing {column}: {list(miss)}')

    if maintain_order:
        _column = '_' + column
        index = pl.DataFrame(
            {column: index},
            schema_overrides={column: df.schema[column]}
        ).with_row_index(_column)
        ret = df.lazy().join(index, on=column, how='left')
        ret = ret.filter(pl.col(_column).is_not_null())
        return ret.sort(_column).drop(_column).collect()
    else:
        return df.filter(pl.col(column).is_in(index))
