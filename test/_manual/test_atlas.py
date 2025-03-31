import unittest
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from neuralib.atlas.cellatlas.core import load_cellatlas
from neuralib.atlas.data import load_structure_tree, load_bg_structure_tree, get_children, get_leaf_in_annotation, \
    build_annotation_leaf_map
from neuralib.atlas.view import load_slice_view
from neuralib.util.dataframe import assert_polars_equal_verbose


@unittest.skip('manually ')
class TestLegacyData(unittest.TestCase):
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


class TestBrainGlobe(unittest.TestCase):

    def test_structure_tree_with_ccf(self):
        cols = ['acronym', 'id']
        x = load_structure_tree().select(cols)
        y = load_bg_structure_tree().select(cols)

        with self.assertRaises(AssertionError):
            assert_polars_equal_verbose(x, y)

    def test_get_child_id(self):
        ret = get_children(385, dataframe=False)  # VISp
        exp = [593, 821, 721, 778, 33, 305]
        self.assertListEqual(ret, exp)

    def test_get_child_acronym(self):
        ret = get_children('VISp', dataframe=False)
        exp = ['VISp1', 'VISp2/3', 'VISp4', 'VISp5', 'VISp6a', 'VISp6b']
        self.assertListEqual(ret, exp)

    def test_get_leaf_in_annotation(self):
        x = get_leaf_in_annotation('VISp', name=True)
        print(x)

    def test_build_all_leaf_map(self):
        x = set(build_annotation_leaf_map()[385])
        y = set(get_leaf_in_annotation('VISp'))
        self.assertSetEqual(x, y)

    def test_volume_dataframe(self):
        ...


class TestView(unittest.TestCase):

    @patch('matplotlib.pyplot.show')
    def test_slice_view_reference(self, *arg):
        from neuralib.atlas.view import load_slice_view

        slice_index = 700
        plane = load_slice_view('reference', plane_type='coronal', resolution=10).plane_at(slice_index)

        _, ax = plt.subplots(ncols=3, figsize=(20, 10))
        plane.plot(ax=ax[0], boundaries=True)
        plane.with_angle_offset(deg_x=15, deg_y=0).plot(ax=ax[1], boundaries=True)
        plane.with_angle_offset(deg_x=0, deg_y=20).plot(ax=ax[2], boundaries=True)
        plt.show()

    @patch('matplotlib.pyplot.show')
    def test_slice_view_annotation(self, *arg):
        from neuralib.atlas.view import load_slice_view

        slice_index = 500
        plane = load_slice_view('annotation', plane_type='sagittal', resolution=10).plane_at(slice_index)

        _, ax = plt.subplots(ncols=3, figsize=(20, 10))
        plane.plot(ax=ax[0])
        plane.with_angle_offset(deg_x=15, deg_y=0).plot(ax=ax[1])
        plane.with_angle_offset(deg_x=0, deg_y=20).plot(ax=ax[2])
        plt.show()

    @patch('matplotlib.pyplot.show')
    def test_annotation_region(self, *arg):
        slice_index = 800
        plane = load_slice_view('reference', plane_type='coronal', resolution=10).plane_at(slice_index)

        _, ax = plt.subplots()
        plane.with_angle_offset(deg_x=15, deg_y=0).plot(ax=ax, annotation_region=['RSP', 'VISp'])

        plt.show()

    @patch('matplotlib.pyplot.show')
    def test_max_projection(self):
        from neuralib.atlas.view import load_slice_view
        view = load_slice_view('reference', plane_type='transverse', resolution=10)

        _, ax = plt.subplots()
        regions = get_children('VIS')
        view.plot_max_projection(ax, annotation_regions=regions)
        plt.show()


class TestCellAtlas(unittest.TestCase):

    def test_cell_atlas_sync(self):
        from neuralib.atlas.cellatlas import CellAtlas
        x = CellAtlas.load_sync_allen_structure_tree()
        y = load_cellatlas()

        assert_polars_equal_verbose(x, y)


if __name__ == '__main__':
    unittest.main()
