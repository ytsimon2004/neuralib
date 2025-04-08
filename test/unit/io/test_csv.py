import tempfile
from pathlib import Path

import polars as pl
from polars.testing import assert_frame_equal

from neuralib.io import csv_header


def test_csv_header():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.csv"
        with csv_header(path, ["id", "value"]) as csv:
            csv(1, 100)
            csv(2, 200)

        df = pl.read_csv(path)
        exp = pl.DataFrame({'id': [1, 2], 'value': [100, 200]})
        assert_frame_equal(df, exp)


def test_csv_append():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.csv"
        with csv_header(path, ["id", "value"]) as csv:
            csv(1, 100)

        with csv_header(path, ["id", "value"], append=True) as csv:
            csv(2, 200)

        df = pl.read_csv(path)
        exp = pl.DataFrame({'id': [1, 2], 'value': [100, 200]})
        assert_frame_equal(df, exp)


def test_csv_continuous_mode():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.csv"
        with csv_header(path, ["id", "value"]) as csv:
            csv(1, 100)

        with csv_header(path, ["id", "value"], append=True, continuous_mode='id') as csv:
            if 1 not in csv:
                csv(2, 200)
            if 3 not in csv:
                csv(3, 300)

        df1 = pl.read_csv(path)
        df2 = pl.DataFrame({'id': [1, 3], 'value': [100, 300]})
        assert_frame_equal(df1, df2)


def test_csv_header_double_quotes():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.csv"
        with csv_header(path, ["id", "value"], quotes_header="value") as csv:
            csv(1, "a,b,c")

        df1 = pl.read_csv(path)
        df2 = pl.DataFrame({'id': [1], 'value': ["a,b,c"]})
        assert_frame_equal(df1, df2)
