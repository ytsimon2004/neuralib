import unittest

import numpy as np
import polars as pl
from polars.testing import assert_frame_equal

from neuralib.io.dataset import load_example_dlc_h5, load_example_dlc_csv


class TestDlc(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.h5 = load_example_dlc_h5()

    def test_dataformat_same_results(self):
        csv = load_example_dlc_csv()
        assert_frame_equal(self.h5.dat, csv.dat)

    def test_global_lh_filter(self):
        dlc = self.h5.with_global_lh_filter(0.99)
        df = dlc.dat
        for j in dlc.joints:
            x = df.filter(pl.col(f'{j}_likelihood') < 0.99).select(f'{j}_x', f'{j}_y').to_numpy()
            self.assertTrue(np.all(np.isnan(x)))

    def test_get_joint(self):
        item = 'Nose'
        joint = self.h5[item]
        self.assertEqual(vars(joint.source), vars(self.h5))
        self.assertEqual(joint.name, item)
        self.assertListEqual(joint.dat.columns, ['x', 'y', 'likelihood'])

    def test_joint_lh_filter(self):
        item = 'Nose'
        joint = self.h5[item].with_lh_filter(0.99)
        x = joint.dat.filter(pl.col('likelihood') < 0.99).select('x', 'y').to_numpy()
        self.assertTrue(np.all(np.isnan(x)))

    def test_interp_gap2d(self):
        pass


if __name__ == '__main__':
    unittest.main()
