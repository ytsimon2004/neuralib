import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from neuralib.atlas.data import load_structure_tree, load_bg_structure_tree
from neuralib.util.dataframe import assert_polars_equal_verbose


class TestBrainView(unittest.TestCase):

    def test_load_ccf_annotation(self):
        from neuralib.atlas.data import load_ccf_annotation
        arr = load_ccf_annotation()

        self.assertEquals(arr.dtype, np.uint16)
        self.assertTupleEqual(arr.shape, (1320, 800, 1140))

    def test_load_ccf_template(self):
        from neuralib.atlas.data import load_ccf_template
        arr = load_ccf_template()

        self.assertEquals(arr.dtype, np.uint16)
        self.assertTupleEqual(arr.shape, (1320, 800, 1140))

    @unittest.skip('run manually, since module "allensdk" build error')
    def test_load_allensdk_annotation(self):
        from neuralib.atlas.data import load_allensdk_annotation
        arr = load_allensdk_annotation()

        self.assertEquals(arr.dtype, np.uint32)
        self.assertTupleEqual(arr.shape, (1320, 800, 1140))

    @patch('matplotlib.pyplot.show')
    def test_slice_view_reference(self, *arg):
        from neuralib.atlas.view import load_slice_view

        slice_index = 700
        plane = load_slice_view('reference', plane_type='coronal', resolution=10).plane_at(slice_index)

        _, ax = plt.subplots(ncols=3, figsize=(20, 10))
        plane.plot(ax=ax[0], with_annotation=True)
        plane.with_angle_offset(deg_x=15, deg_y=0).plot(ax=ax[1], with_annotation=True)
        plane.with_angle_offset(deg_x=0, deg_y=20).plot(ax=ax[2], with_annotation=True)
        plt.show()

    def test_slice_view_annotation(self, *arg):
        from neuralib.atlas.view import load_slice_view

        slice_index = 500
        plane = load_slice_view('annotation', plane_type='sagittal', resolution=10).plane_at(slice_index)

        _, ax = plt.subplots(ncols=3, figsize=(20, 10))
        plane.plot(ax=ax[0], with_annotation=True)
        plane.with_angle_offset(deg_x=15, deg_y=0).plot(ax=ax[1], with_annotation=True)
        plane.with_angle_offset(deg_x=0, deg_y=20).plot(ax=ax[2], with_annotation=True)
        plt.show()


class TestStructureTree(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.legacy_df = load_structure_tree()
        cls.bg_df = load_bg_structure_tree()

    def test_basic_field(self):
        cols = ['acronym', 'id']
        x = self.legacy_df.select(cols)
        y = self.bg_df.select(cols)

        with self.assertRaises(AssertionError):
            assert_polars_equal_verbose(x, y)


if __name__ == '__main__':
    unittest.main()
