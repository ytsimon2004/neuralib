import math
import unittest
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from numpy.testing import assert_array_equal
from polars.testing import assert_frame_equal

from neuralib.io import pyh5


class GroupData(pyh5.H5pyData):
    e: float = pyh5.attr()


class ReadData(pyh5.H5pyData):
    a: int = pyh5.attr()
    b: int = pyh5.attr()
    c: float = pyh5.attr()
    d: np.ndarray = pyh5.array()
    g: GroupData = pyh5.group()


class TestH5pyDataWrapperRead(unittest.TestCase):
    TMP_FILE = 'TestH5pyDataWrapperRead.h5'

    @classmethod
    def setUpClass(cls):
        f = h5py.File(cls.TMP_FILE, 'w')
        f.attrs['a'] = 1
        f.attrs['b'] = 2
        f.attrs['c'] = 3.141592653589793
        f.create_dataset('d', data=np.arange(12))
        g = f.create_group('g')
        g.attrs['e'] = 2.718281828459045
        f.close()

    @classmethod
    def tearDownClass(cls):
        Path(cls.TMP_FILE).unlink(missing_ok=True)

    DATA: ReadData

    def setUp(self):
        self.DATA = ReadData(self.TMP_FILE)

    def tearDown(self):
        self.DATA = None

    def test_attr(self):
        self.assertEqual(self.DATA.a, 1)
        self.assertEqual(self.DATA.b, 2)
        self.assertAlmostEqual(self.DATA.c, math.pi)

    def test_array(self):
        self.assertEqual(self.DATA.d.ndim, 1)
        self.assertEqual(self.DATA.d.shape, (12,))
        assert_array_equal(self.DATA.d, np.arange(12))

    def test_array_like(self):
        self.assertEqual(self.DATA.d[2], np.arange(12)[2])
        self.assertAlmostEqual(np.mean(self.DATA.d), np.mean(np.arange(12)))

    def test_group(self):
        g = self.DATA.g
        self.assertIsInstance(g, GroupData)
        self.assertAlmostEqual(g.e, math.e)

    def test_set_attr(self):
        with self.assertRaises(KeyError):
            self.DATA.a = 0

    def test_set_array(self):
        with self.assertRaises(KeyError):
            self.DATA.d = np.arange(24)


class TestH5pyDataWrapperWrite(unittest.TestCase):
    def tearDown(self):
        Path(TestH5pyDataWrapperRead.TMP_FILE).unlink(missing_ok=True)

    def test_write(self):
        data = ReadData(TestH5pyDataWrapperRead.TMP_FILE, 'w')
        data.a = 1
        data.b = 2
        data.c = 3.141592653589793
        data.d = np.arange(12)
        data.g.e = 2.718281828459045
        data = None

        main = TestH5pyDataWrapperRead()
        main.DATA = ReadData(TestH5pyDataWrapperRead.TMP_FILE)
        main.test_attr()
        main.test_array()
        main.test_array_like()
        main.test_group()
        main.test_set_attr()
        main.test_set_array()


class TestH5pyDataWrapperTable(unittest.TestCase):
    TMP_FILE = 'TestH5pyDataWrapperTable.h5'

    def tearDown(self):
        Path(self.TMP_FILE).unlink(missing_ok=True)

    def test_read_write_table_default(self):
        class Test(pyh5.H5pyData):
            df: pl.DataFrame = pyh5.table()

        df = pl.DataFrame(data=dict(
            a=[0, 1, 2, 3],
            b=[0.0, 0.1, 1.0, 10.0],
            c=['a', 'b', 'c', 'd']
        ))

        test = Test(self.TMP_FILE, 'w')
        test.df = df
        test = None

        test = Test(self.TMP_FILE, 'r')
        ret = test.df

        assert_frame_equal(df, ret)

    @unittest.skip('TODO test_read_write_table_pytables')
    def test_read_write_table_pytables(self):
        pass


if __name__ == '__main__':
    unittest.main()
