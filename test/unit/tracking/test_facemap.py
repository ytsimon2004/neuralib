from unittest.mock import patch

import numpy as np
import pytest

from neuralib.io.dataset import load_example_facemap_pupil, load_example_facemap_keypoints
from neuralib.tracking.facemap import FaceMapResult
from neuralib.tracking.facemap.plot import plot_cmap_time_series


# ---- Pupil ---- #

@pytest.fixture(scope='module')
def pupil() -> FaceMapResult:
    return load_example_facemap_pupil()


def test_get_pupil(pupil: FaceMapResult):
    assert isinstance(pupil.get_pupil(), dict)


def test_get_pupil_location_movement(pupil: FaceMapResult):
    pupil.get_pupil_location_movement()


def test_get_blink(pupil: FaceMapResult):
    pupil.get_blink()


# ---- KeyPoint ---- #

@pytest.fixture(scope='module')
def keypoint() -> FaceMapResult:
    return load_example_facemap_keypoints(cached=True, rename_folder='facemap_kp')


def test_with_keypoint(keypoint):
    assert keypoint.with_keypoint


def test_keypoints(keypoint: FaceMapResult):
    assert keypoint.keypoints == [
        'eye(back)', 'eye(bottom)', 'eye(front)', 'eye(top)', 'lowerlip', 'mouth',
        'nose(bottom)', 'nose(r)', 'nose(tip)', 'nose(top)', 'nosebridge',
        'paw', 'whisker(I)', 'whisker(II)', 'whisker(III)'
    ]


def test_get_single_keypoint(keypoint: FaceMapResult):
    df = keypoint.get('eye(back)')
    assert df['keypoint'].unique().item() == 'eye(back)'


def test_get_multiple_keypoints(keypoint: FaceMapResult):
    df = keypoint.get('eye(back)', 'mouth')
    assert df['keypoint'].n_unique() == 2


@patch("matplotlib.pyplot.show")
def test_plot_keypoints(mock_show, keypoint: FaceMapResult):
    from neuralib.tracking.facemap.plot import plot_facemap_keypoints
    plot_facemap_keypoints(keypoint, frame_interval=(0, 500))


@patch("matplotlib.pyplot.show")
def test_plot_multiple_keypoints(mock_show, keypoint: FaceMapResult):
    df = keypoint.get('eye(back)').with_outlier_filter().to_zscore()
    x = np.array(df['x'])
    y = np.array(df['y'])
    plot_cmap_time_series(x, y)
