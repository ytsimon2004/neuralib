import unittest
from pathlib import Path
from unittest.mock import patch

import pytest
from matplotlib import pyplot as plt

from neuralib.atlas.cellatlas.core import load_cellatlas
from neuralib.atlas.data import get_children, get_leaf_in_annotation, build_annotation_leaf_map, load_bg_volumes
from neuralib.atlas.view import get_slice_view

DATA_EXISTS = (Path().home() / '.brainglobe' / 'allen_mouse_100um_v1.2').exists()


@pytest.mark.skipif(not DATA_EXISTS, reason="source data need to be downloaded")
class TestBrainGlobe(unittest.TestCase):

    def test_get_child_id(self):
        ret = get_children(385, dataframe=False, atlas_name='allen_mouse_100um')  # VISp
        exp = [593, 821, 721, 778, 33, 305]
        self.assertListEqual(ret, exp)

    def test_get_child_acronym(self):
        ret = get_children('VISp', dataframe=False, atlas_name='allen_mouse_100um')
        exp = ['VISp1', 'VISp2/3', 'VISp4', 'VISp5', 'VISp6a', 'VISp6b']
        self.assertListEqual(ret, exp)

    def test_get_leaf_in_annotation(self):
        x = get_leaf_in_annotation('VISp', name=True, atlas_name='allen_mouse_100um')
        print(x)

    def test_build_all_leaf_map(self):
        x = set(build_annotation_leaf_map(atlas_name='allen_mouse_100um')[385])
        y = set(get_leaf_in_annotation('VISp'))
        self.assertSetEqual(x, y)


@pytest.mark.skipif(not DATA_EXISTS, reason="source data need to be downloaded")
class TestView(unittest.TestCase):

    @patch('matplotlib.pyplot.show')
    def test_slice_view_reference(self, *arg):
        from neuralib.atlas.view import get_slice_view

        slice_index = 30
        plane = get_slice_view('reference', plane_type='coronal', resolution=100).plane_at(slice_index)

        _, ax = plt.subplots(ncols=3, figsize=(20, 10))
        plane.plot(ax=ax[0], boundaries=True)
        plane.with_angle_offset(deg_x=15, deg_y=0).plot(ax=ax[1], boundaries=True)
        plane.with_angle_offset(deg_x=0, deg_y=20).plot(ax=ax[2], boundaries=True)
        plt.show()

    @patch('matplotlib.pyplot.show')
    def test_slice_view_annotation(self, *arg):
        from neuralib.atlas.view import get_slice_view

        slice_index = 500
        plane = get_slice_view('annotation', plane_type='sagittal', resolution=100).plane_at(slice_index)

        _, ax = plt.subplots(ncols=3, figsize=(20, 10))
        plane.plot(ax=ax[0])
        plane.with_angle_offset(deg_x=15, deg_y=0).plot(ax=ax[1])
        plane.with_angle_offset(deg_x=0, deg_y=20).plot(ax=ax[2])
        plt.show()

    @patch('matplotlib.pyplot.show')
    def test_annotation_region(self, *arg):
        slice_index = 800
        plane = get_slice_view('reference', plane_type='coronal', resolution=100).plane_at(slice_index)

        _, ax = plt.subplots()
        plane.with_angle_offset(deg_x=15, deg_y=0).plot(ax=ax, annotation_region=['RSP', 'VISp'])

        plt.show()

    @patch('matplotlib.pyplot.show')
    def test_affine_transform(self, *args):
        import matplotlib.transforms as mtransforms

        slice_index = 800
        plane = get_slice_view('reference', plane_type='coronal', resolution=100).plane_at(slice_index)

        _, ax = plt.subplots()
        aff = mtransforms.Affine2D().skew_deg(-20, 0)
        t = aff + ax.transData

        plane.with_angle_offset().plot(ax=ax, transform=t)
        plt.show()

    @patch('matplotlib.pyplot.show')
    def test_max_projection(self, *args):
        from neuralib.atlas.view import get_slice_view
        view = get_slice_view('reference', plane_type='transverse', resolution=100)

        _, ax = plt.subplots()
        regions = get_children('VIS')
        view.plot_max_projection(ax, annotation_regions=regions)
        plt.show()


@pytest.mark.skipif(not DATA_EXISTS, reason="source data need to be downloaded")
class TestRegionVolumes(unittest.TestCase):

    @patch('matplotlib.pyplot.show')
    def test_volume_different_source_data(self, *args):
        x = load_bg_volumes('allen_mouse_100um')
        y = load_cellatlas().select('acronym', 'Volumes [mm^3]')

        z = x.join(y, on='acronym')
        cols = z['volume_mm3', 'Volumes [mm^3]'].to_numpy()
        plt.plot(cols[:, 0], cols[:, 1], 'k.')
        plt.show()


if __name__ == '__main__':
    unittest.main()
