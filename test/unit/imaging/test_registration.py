import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.testing import assert_array_equal

from neuralib.imaging.registration.coordinates import get_field_of_view, get_cellular_coordinate


@pytest.fixture(scope='module')
def fov():
    am = [0, -1.17]
    pm = [0, -2.17]
    pl = [-0.67, -2.17]
    al = [-0.67, -1.17]
    return get_field_of_view(am, pm, pl, al)


def test_fov_perpendicular(fov):
    assert fov.perpendicular
    patch = fov.to_polygon()
    fig, ax = plt.subplots()
    ax.add_patch(patch)
    ax.set_aspect('equal')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-3, 1)


def test_cellular_relative_origin(fov):
    neuron_idx = np.arange(3)
    ap = np.array([-1, -2, -3])
    ml = np.array([0, -1, -2])
    coords = get_cellular_coordinate(neuron_idx, ap, ml)
    assert coords.unit == 'mm'
    assert_array_equal(coords.plane_index, np.full_like(coords.neuron_idx, 0))
    #
    new_coords = coords.relative_origin(fov, origin='am')
    assert new_coords.unit == 'um'
    assert_array_equal(new_coords.ap, -1.17 * 1000 - ap * 1000)
    assert_array_equal(new_coords.ml, -ml * 1000)


def test_cellular_set_attrs():
    neuron_idx = np.arange(3)
    ap = np.array([-1, -2, -3])
    ml = np.array([0, -1, -2])
    coords = get_cellular_coordinate(neuron_idx, ap, ml)

    assert coords.with_masking(np.array([0, 1, 1], dtype=bool)).ap.shape == (2,)
    assert_array_equal(coords.with_value(np.arange(3)).value, np.arange(3))
