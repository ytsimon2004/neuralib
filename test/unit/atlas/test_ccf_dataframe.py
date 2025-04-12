from pathlib import Path

import pytest

from neuralib.atlas.ccf import RoiClassifierDataFrame
from neuralib.io.dataset import load_example_rois

DATA_EXISTS = (Path().home() / '.brainglobe' / 'allen_mouse_10um_v1.2').exists()


@pytest.fixture(scope='module', autouse=True)
def roi() -> RoiClassifierDataFrame:
    return RoiClassifierDataFrame(load_example_rois(cached=True, rename_file='roi')).post_processing()


def test_post_processing(roi: RoiClassifierDataFrame):
    post_columns = ['tree_0', 'tree_1', 'tree_2', 'tree_3', 'tree_4', 'family']
    cols = roi.columns
    for c in post_columns:
        assert c in cols


def test_to_normalized(roi: RoiClassifierDataFrame):
    roi.to_normalized(norm='channel', level=3)


@pytest.mark.skipif(not DATA_EXISTS, reason='not existed cache')
def test_to_normalized_bg(roi: RoiClassifierDataFrame):
    roi.to_normalized(norm='volume', level=1)
    roi.to_normalized(norm='cell', level=2)


def test_normalized_transform(roi: RoiClassifierDataFrame):
    norm = roi.to_normalized(norm='channel', level=2)
    norm.to_winner(['aRSC', 'pRSC'])
    norm.to_bias_index('aRSC', 'pRSC')


def test_to_subregion(roi: RoiClassifierDataFrame):
    roi.to_subregion('VIS')
