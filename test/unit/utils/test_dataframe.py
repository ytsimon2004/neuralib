import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from neuralib.util.dataframe import DataFrameWrapper, helper_with_index_column


class MyDataFrame(DataFrameWrapper):
    def __init__(self, data: pl.DataFrame):
        self._data = data

    def dataframe(self, dataframe: pl.DataFrame = None, may_inplace=True):
        if dataframe is None:
            return self._data
        if may_inplace:
            self._data = dataframe
            return self
        else:
            return MyDataFrame(dataframe)


@pytest.fixture
def df() -> MyDataFrame:
    return MyDataFrame(pl.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]}))


def test_len_and_columns(df):
    assert len(df) == 3
    assert df.columns == ['a', 'b']


def test_filter(df):
    table = df.clone()
    filtered = table.filter(pl.col('a') > 1)
    expected = pl.DataFrame({'a': [2, 3], 'b': [20, 30]})
    assert_frame_equal(filtered.dataframe(), expected)


def test_sort(df):
    table = df.clone()
    unsorted_df = pl.DataFrame({'a': [3, 1, 2], 'b': [30, 10, 20]})
    table.dataframe(unsorted_df, may_inplace=True)
    sorted_table = table.sort('a')
    expected = pl.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30]})
    assert_frame_equal(sorted_table.dataframe(), expected)


def test_rename(df):
    table = df.clone()
    renamed = table.rename({'b': 'B'})
    expected = pl.DataFrame({'a': [1, 2, 3], 'B': [10, 20, 30]})
    assert_frame_equal(renamed.dataframe(), expected)


def test_with_columns(df):
    table = df.clone()
    new_table = table.with_columns((pl.col('a') * 10).alias('b_new'))
    expected = pl.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30], 'b_new': [10, 20, 30]})
    assert_frame_equal(new_table.dataframe(), expected)


def test_lazy_wrapper(df):
    table = df.clone()
    lazy_wrapper = table.lazy().filter(pl.col('a') > 1).rename({'b': 'B'})
    collected = lazy_wrapper.collect()
    expected = pl.DataFrame({'a': [2, 3], 'B': [20, 30]})
    assert_frame_equal(collected.dataframe(), expected)


def test_join(df: MyDataFrame):
    table = df.clone()
    df_right = pl.DataFrame({'a': [2, 3], 'c': [200, 300]})
    table_join = table.join(df_right, on='a', how='inner')
    expected = pl.DataFrame({'a': [2, 3], 'b': [20, 30], 'c': [200, 300]})
    assert_frame_equal(table_join.dataframe(), expected)


def test_partition_by(df):
    table = df.clone()
    df = pl.DataFrame({
        'group': ['x', 'y', 'x', 'y'],
        'a': [1, 2, 3, 4]
    })
    table.dataframe(df, may_inplace=True)
    parts = table.partition_by('group', as_dict=True)
    assert isinstance(parts, dict)
    assert set(parts.keys()) == {('x',), ('y',)}
    expected_x = df.filter(pl.col('group') == 'x')
    assert_frame_equal(parts[('x',)].dataframe(), expected_x)


def test_pipe(df):
    table = df.clone()

    def add_column(df: pl.DataFrame, multiplier: int) -> pl.DataFrame:
        return df.with_columns((pl.col('a') * multiplier).alias('b_new'))

    t2 = table.pipe(add_column, 5)
    expected = pl.DataFrame({'a': [1, 2, 3], 'b': [10, 20, 30], 'b_new': [5, 10, 15]})
    assert_frame_equal(t2.dataframe(), expected)


def test_getitem(df):
    table = df.clone()
    col_a = table['a']
    expected_series = table.dataframe()['a']
    assert_series_equal(col_a, expected_series)

    sub_df = table[('a', 'b')]
    assert_frame_equal(sub_df, table.dataframe())


def test_helper_with_index_column():
    df = pl.DataFrame({'id': [1, 2, 3, 4], 'value': [10, 20, 30, 40]})
    table = MyDataFrame(df)

    # non-strict filtering.
    filtered = helper_with_index_column(table, 'id', [2, 3], maintain_order=False, strict=False)
    expected = df.filter(pl.col('id').is_in([2, 3]))
    assert_frame_equal(filtered.dataframe(), expected)

    # strict mode
    with pytest.raises(RuntimeError):
        helper_with_index_column(table, 'id', [2, 5], strict=True)
