import polars as pl
import pytest

from neuralib.atlas.ccf import RoiClassifierDataFrame
from neuralib.io.dataset import load_example_rois


@pytest.fixture
def roi() -> pl.DataFrame:
    return load_example_rois(cached=True, rename_file='roi')


def test_post_processing(roi: pl.DataFrame):
    post_columns = ['tree_0', 'tree_1', 'tree_2', 'tree_3', 'tree_4', 'family']
    cols = RoiClassifierDataFrame(roi).post_processing().columns
    for c in post_columns:
        assert c in cols


def test_to_normalized(roi: pl.DataFrame):
    post = RoiClassifierDataFrame(roi).post_processing()
    post.to_normalized(norm='volume', level=1)
    post.to_normalized(norm='cell', level=2)
    post.to_normalized(norm='channel', level=3)


def test_normalized_transform(roi: pl.DataFrame):
    norm = RoiClassifierDataFrame(roi).post_processing().to_normalized(norm='volume', level=2)
    norm.to_winner(['aRSC', 'pRSC'])
    norm.to_bias_index('aRSC', 'pRSC')


def test_to_subregion(roi: pl.DataFrame):
    RoiClassifierDataFrame(roi).post_processing().to_subregion('VIS')
