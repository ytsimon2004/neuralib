import polars as pl
from polars.testing import assert_frame_equal

from neuralib.util.verbose import printdf

__all__ = ['assert_polars_equal_verbose']


def assert_polars_equal_verbose(df1: pl.DataFrame, df2: pl.DataFrame):
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
            printdf(df1_extra)

        if df2_extra.height > 0:
            print('\nRows in df2 not in df1:')
            printdf(df2_extra)

        # If shapes match, show cell-wise diff
        if df1.shape == df2.shape:
            print('\nCell-wise differences (non-equal values):')
            diffs = _highlight_cell_differences(df1, df2)
            print(diffs)

        raise e


def _highlight_cell_differences(df1: pl.DataFrame, df2: pl.DataFrame) -> pl.DataFrame:
    return pl.DataFrame({
        col: df1[col].cast(str).zip_with(
            df1[col] != df2[col],
            pl.lit('df1=') + df1[col].cast(str) + ', df2=' + df2[col].cast(str)
        ).fill_null('')  # Handle NaNs
        for col in df1.columns
    })
