import unittest

import numpy as np
import polars as pl
from numpy.testing import assert_array_equal

from neuralib.util.util_polars import DataFrameWrapper, helper_with_index_column


class SimpleDataFrameWrapper(DataFrameWrapper):
    def __init__(self, dataframe: pl.DataFrame):
        self._dataframe = dataframe

    def dataframe(self, dataframe: pl.DataFrame = None, may_inplace=True):
        if dataframe is None:
            return self._dataframe
        else:
            return SimpleDataFrameWrapper(dataframe)


class TestPolarHelperFunctions(unittest.TestCase):
    def test_helper_with_index_column(self):
        df = SimpleDataFrameWrapper(pl.DataFrame({'index': np.arange(0, 10, 2)}))

        with self.subTest('test_single_index'):
            ret = helper_with_index_column(df, 'index', 2)
            assert_array_equal(ret['index'].to_numpy(), np.array([2]))

        with self.subTest('test_single_index_miss'):
            ret = helper_with_index_column(df, 'index', 3)
            assert_array_equal(ret['index'].to_numpy(), np.array([]))

        with self.subTest('test_single_index_miss_strict'):
            with self.assertRaises(RuntimeError):
                helper_with_index_column(df, 'index', 3, strict=True)

        with self.subTest('test_list_index'):
            ret = helper_with_index_column(df, 'index', [2, 4])
            assert_array_equal(ret['index'].to_numpy(), np.array([2, 4]))

        with self.subTest('test_list_index_miss'):
            ret = helper_with_index_column(df, 'index', [2, 3, 4])
            assert_array_equal(ret['index'].to_numpy(), np.array([2, 4]))

        with self.subTest('test_list_index_miss_strict'):
            with self.assertRaises(RuntimeError):
                helper_with_index_column(df, 'index', [2, 3, 4], strict=True)

        with self.subTest('test_numpy_index'):
            ret = helper_with_index_column(df, 'index', np.array([2, 4]))
            assert_array_equal(ret['index'].to_numpy(), np.array([2, 4]))

        with self.subTest('test_dataframe_index'):
            ref = helper_with_index_column(df, 'index', np.array([2, 4]))
            ret = helper_with_index_column(df, 'index', ref)
            assert_array_equal(ret['index'].to_numpy(), np.array([2, 4]))

        with self.subTest('test_list_index_wo_maintain_order'):
            ret = helper_with_index_column(df, 'index', np.array([4, 2]))
            assert_array_equal(ret['index'].to_numpy(), np.array([2, 4]))
            ret = helper_with_index_column(df, 'index', np.array([4, 2]), maintain_order=True)
            assert_array_equal(ret['index'].to_numpy(), np.array([4, 2]))
            ret = helper_with_index_column(df, 'index', np.array([4, 3, 2]), maintain_order=True)
            assert_array_equal(ret['index'].to_numpy(), np.array([4, 2]))

if __name__ == '__main__':
    unittest.main()
