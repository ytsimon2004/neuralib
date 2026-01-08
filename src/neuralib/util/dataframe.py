from __future__ import annotations

import abc
import numpy as np
import polars as pl
from polars.dataframe.group_by import GroupBy
from polars.testing import assert_frame_equal
from typing import Callable, Literal, overload, TypeVar, Generic, Any, TYPE_CHECKING

from neuralib.util.verbose import printdf

if TYPE_CHECKING:
    from collections.abc import Sequence, Iterable, Collection
    from typing import Self, ParamSpec, Concatenate
    from polars import _typing as pty

    P = ParamSpec('P')

__all__ = ['DataFrameWrapper',
           'helper_with_index_column',
           'assert_polars_equal_verbose']


class DataFrameWrapper(metaclass=abc.ABCMeta):
    """
    Abstract wrapper class for a `polars.DataFrame`, enabling convenient and composable
    dataframe operations in a subclassable, object-oriented interface.

    This base class is intended to be inherited by custom data structures whose core data
    is represented as a `polars.DataFrame`. It provides a suite of standard dataframe
    operations (e.g., filtering, sorting, renaming, joining) that return the wrapper
    instance (`Self`), preserving method chaining and encapsulation.

    This allows users to write clean, expressive logic using their custom wrapper class
    while still leveraging the full power of Polars.

    Subclasses **must** implement the `dataframe` method to get or set the internal
    `polars.DataFrame`.

    Examples
    --------
    A minimal subclass that wraps a Polars DataFrame:

    >>> class MyTable(DataFrameWrapper):
    ...     def __init__(self, data: pl.DataFrame):
    ...         self._data = data
    ...
    ...     def dataframe(self, dataframe: pl.DataFrame = None, may_inplace=True):
    ...         if dataframe is None:
    ...             return self._data
    ...         if may_inplace:
    ...             self._data = dataframe
    ...             return self
    ...         else:
    ...             return MyTable(dataframe)

    >>> df = pl.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
    >>> t = MyTable(df)
    >>> t = t.filter(pl.col("a") > 1).rename({"b": "B"})
    >>> print(t.dataframe())
    shape: (2, 2)
    ┌─────┬─────┐
    │ a   │ B   │
    ├─────┼─────┤
    │ 2   │ 20  │
    │ 3   │ 30  │
    └─────┴─────┘

    Notes
    -----
    - All supported operations delegate to the underlying `polars.DataFrame` and return
      the modified wrapper instance.
    - The actual `dataframe` storage and logic is delegated to subclasses via the abstract
      `dataframe()` getter/setter method.
    - This class is designed for flexible and extensible use in applications such as
      data modeling, pipelines, or typed schema handling.

    Supported Operations
    --------------------
    - Accessors: `columns`, `schema`, `__len__`, `__array__`, `__dataframe__`
    - Indexing: `__getitem__`
    - Structure: `filter`, `drop`, `drop_nulls`, `fill_null`, `fill_nan`, `select`,
                 `with_columns`, `with_row_index`, `rename`, `slice`, `head`, `tail`, `limit`, `sort`
    - Aggregation: `group_by`
    - Partitioning: `partition_by`
    - Joining: `join`
    - Transformation: `pipe`, `clone`, `lazy`

    See Also
    --------
    polars.DataFrame : The underlying DataFrame API
    polars.Expr : Expression system used throughout the API
    """

    @overload
    def dataframe(self) -> pl.DataFrame:
        pass

    @overload
    def dataframe(self, dataframe: pl.DataFrame, may_inplace=True) -> Self:
        pass

    @abc.abstractmethod
    def dataframe(self, dataframe: pl.DataFrame = None, may_inplace=True):
        """
        Getter/setter for the internal Polars DataFrame.

        :param dataframe: Optional new dataframe to set.
        :param may_inplace: If True, update current instance. Otherwise, return new instance.
        :return: The current dataframe or a modified wrapper instance.
        """
        pass

    def __len__(self) -> int:
        """See `polars.DataFrame.__len__`."""
        return len(self.dataframe())

    @property
    def columns(self) -> list[str]:
        """See `polars.DataFrame.columns`."""
        return self.dataframe().columns

    @property
    def schema(self) -> pl.Schema:
        """See `polars.DataFrame.schema`."""
        return self.dataframe().schema

    def __array__(self, *args, **kwargs) -> np.ndarray:
        """See `polars.DataFrame.__array__`."""
        return self.dataframe().__array__(*args, **kwargs)

    def __dataframe__(self, *args, **kwargs):
        """See `polars.DataFrame.__dataframe__`."""
        return self.dataframe().__dataframe__(*args, **kwargs)

    @overload
    def __getitem__(self, key: (
            str
            | tuple[[pty.MultiIndexSelector, pty.SingleColSelector]]
    )) -> pl.Series:
        ...

    @overload
    def __getitem__(self, key: (
            pty.SingleIndexSelector
            | pty.MultiIndexSelector
            | pty.MultiColSelector
            | tuple[pty.SingleIndexSelector, pty.MultiColSelector]
            | tuple[pty.MultiIndexSelector, pty.MultiColSelector]
    )) -> pl.DataFrame:
        ...

    def __getitem__(self, item):
        """See `polars.DataFrame.__getitem__`."""
        return self.dataframe().__getitem__(item)

    def lazy(self) -> LazyDataFrameWrapper[Self]:
        """Wrap dataframe in a lazy wrapper."""
        return LazyDataFrameWrapper(self, self.dataframe().lazy())

    def rename(self, mapping: dict[str, str] | Callable[[str], str]) -> Self:
        """See `polars.DataFrame.rename`."""
        return self.dataframe(self.dataframe().rename(mapping))

    def filter(self, *predicates: pty.IntoExprColumn | Iterable[pty.IntoExprColumn] | bool | list[bool] | np.ndarray,
               **constraints: Any) -> Self:
        """See `polars.DataFrame.filter`."""
        return self.dataframe(self.dataframe().filter(*predicates, **constraints))

    def slice(self, offset: int, length: int | None = None) -> Self:
        """See `polars.DataFrame.slice`."""
        return self.dataframe(self.dataframe().slice(offset, length))

    def head(self, n: int = 5) -> Self:
        """See `polars.DataFrame.head`."""
        return self.dataframe(self.dataframe().head(n))

    def tail(self, n: int = 5) -> Self:
        """See `polars.DataFrame.tail`."""
        return self.dataframe(self.dataframe().tail(n))

    def limit(self, n: int = 5) -> Self:
        """See `polars.DataFrame.limit`."""
        return self.dataframe(self.dataframe().limit(n))

    @overload
    def sort(self,
             by: pty.IntoExpr | Iterable[[pty.IntoExpr]],
             *more_by: pty.IntoExpr,
             descending: bool | Sequence[bool] = False,
             nulls_last: bool | Sequence[bool] = False,
             multithreaded: bool = True,
             maintain_order: bool = False) -> Self:
        ...

    def sort(self, by, *more_by, **kwargs) -> Self:
        """See `polars.DataFrame.sort`."""
        return self.dataframe(self.dataframe().sort(by, *more_by, **kwargs))

    def drop(self, *columns: pty.ColumnNameOrSelector | Iterable[pty.ColumnNameOrSelector],
             strict: bool = True) -> Self:
        """See `polars.DataFrame.drop`."""
        return self.dataframe(self.dataframe().drop(*columns, strict=strict))

    def drop_nulls(self, subset: pty.ColumnNameOrSelector | Collection[pty.ColumnNameOrSelector]) -> Self:
        """See `polars.DataFrame.drop_nulls`."""
        return self.dataframe(self.dataframe().drop_nulls(subset))

    def fill_null(self, value: Any | pl.Expr | None = None,
                  strategy: pty.FillNullStrategy | None = None,
                  limit: int | None = None, **kwargs) -> Self:
        """See `polars.DataFrame.fill_null`."""
        return self.dataframe(self.dataframe().fill_null(value, strategy, limit, **kwargs))

    def fill_nan(self, value: pl.Expr | int | float | None = None) -> Self:
        """See `polars.DataFrame.fill_nan`."""
        return self.dataframe(self.dataframe().fill_nan(value))

    def clear(self, n: int = 5) -> Self:
        """See `polars.DataFrame.clear`."""
        return self.dataframe(self.dataframe().clear(n))

    def clone(self) -> Self:
        """Clone the wrapper."""
        return self.dataframe(self.dataframe(), may_inplace=False)

    @overload
    def partition_by(self, by: pty.ColumnNameOrSelector | Iterable[pty.ColumnNameOrSelector],
                     *more_by: pty.ColumnNameOrSelector,
                     maintain_order: bool = True,
                     include_key: bool = True,
                     as_dict: Literal[False] = ...) -> list[Self]:
        ...

    @overload
    def partition_by(self, by: pty.ColumnNameOrSelector | Iterable[pty.ColumnNameOrSelector],
                     *more_by: pty.ColumnNameOrSelector,
                     maintain_order: bool = ...,
                     include_key: bool = ...,
                     as_dict: Literal[True]) -> dict[tuple[object, ...], Self]:
        ...

    def partition_by(self, by, *more_by, as_dict=False, **kwargs):
        """See `polars.DataFrame.partition_by`."""
        dataframe = self.dataframe().partition_by(by, *more_by, as_dict=as_dict, **kwargs)
        if as_dict:
            return {k: self.dataframe(it, may_inplace=False) for k, it in dataframe.items()}
        else:
            return [self.dataframe(it, may_inplace=False) for it in dataframe]

    def select(self, *exprs: pty.IntoExpr | Iterable[pty.IntoExpr],
               **named_exprs: pty.IntoExpr) -> Self:
        """See `polars.DataFrame.select`."""
        return self.dataframe(self.dataframe().select(*exprs, **named_exprs))

    def with_columns(self, *exprs: pty.IntoExpr | Iterable[pty.IntoExpr],
                     **named_exprs: pty.IntoExpr) -> Self:
        """See `polars.DataFrame.with_columns`."""
        return self.dataframe(self.dataframe().with_columns(*exprs, **named_exprs))

    def with_row_index(self, name: str = "index", offset: int = 0) -> Self:
        """See `polars.DataFrame.with_row_index`."""
        return self.dataframe(self.dataframe().with_row_index(name, offset))

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
        ...

    def join(self, other: pl.DataFrame | DataFrameWrapper, on, *args, **kwargs) -> Self:
        """See `polars.DataFrame.join`."""
        if isinstance(other, DataFrameWrapper):
            other = other.dataframe()
        return self.dataframe(self.dataframe().join(other, on, *args, **kwargs))

    def pipe(self, function: Callable[Concatenate[pl.DataFrame, P], pl.DataFrame],
             *args: P.args,
             **kwargs: P.kwargs) -> Self:
        """See `polars.DataFrame.pipe`."""
        return self.dataframe(self.dataframe().pipe(function, *args, **kwargs))

    def group_by(self, *by: pty.IntoExpr | Iterable[pty.IntoExpr],
                 maintain_order: bool = False,
                 **named_by: pty.IntoExpr) -> GroupBy:
        """See `polars.DataFrame.group_by`."""
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


def assert_polars_equal_verbose(df1: pl.DataFrame, df2: pl.DataFrame, **kwargs):
    """
    Assert that two Polars DataFrames are equal and provide detailed diagnostics if they differ

    :param df1: The first Polars DataFrame to compare
    :param df2: The second Polars DataFrame to compare
    :param kwargs: keyword arguments passed to :func:`~neuralib.util.verbose.printdf()`
    :return:
    """
    try:
        assert_frame_equal(df1, df2)
        print('DataFrames are equal.')
    except AssertionError as e:
        print('DataFrames are NOT equal.')

        # shape
        print('\nShape mismatch:')
        print(f'df1: {df1.shape}')
        print(f'df2: {df2.shape}')

        # column
        if df1.columns != df2.columns:
            print('\nColumn mismatch:')
            print(f'df1 columns: {df1.columns}')
            print(f'df2 columns: {df2.columns}')
            raise e

        df1_extra = df1.join(df2, on=df1.columns, how='anti')
        df2_extra = df2.join(df1, on=df1.columns, how='anti')

        if df1_extra.height > 0:
            print('\nRows in df1 not in df2:')
            printdf(df1_extra, **kwargs)

        if df2_extra.height > 0:
            print('\nRows in df2 not in df1:')
            printdf(df2_extra, **kwargs)

        # If shapes match, show cell-wise diff
        if df1.shape == df2.shape:
            print('\nCell-wise differences (non-equal values):')
            diffs = _highlight_cell_differences(df1, df2)
            printdf(diffs, **kwargs)

        raise e


def _highlight_cell_differences(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({
        col: df1[col].cast(str).zip_with(
            df1[col] != df2[col],
            pl.lit('df1=') + df1[col].cast(str) + ', df2=' + df2[col].cast(str)
        ).fill_null('')  # Handle NaNs
        for col in df1.columns
    })
