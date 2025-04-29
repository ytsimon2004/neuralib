from unittest.mock import patch

import pytest
from matplotlib import pyplot as plt

from neuralib.io import NEUROLIB_DATASET_DIRECTORY
from neuralib.io.dataset import load_example_czi
from neuralib.scan.czi import czi_file

DATA = NEUROLIB_DATASET_DIRECTORY / 'test.czi'
DOWNLOAD_CACHE = False

if not DATA.exists() and DOWNLOAD_CACHE:
    load_example_czi(cached=True)


@pytest.fixture(scope='module')
def czi():
    with czi_file(DATA) as czi:
        yield czi


@pytest.mark.skipif(not DATA.exists(), reason='no cached data')
def test_config(czi):
    assert czi.consistent_config
    assert czi.is_mosaic


@pytest.mark.skipif(not DATA.exists(), reason='no cached data')
def test_dimcode(czi):
    assert czi.dimcode == 'HSTCZMYX'
    assert czi.get_code(0, 'C') == 3
    assert czi.get_code(0, 'M') == 54
    assert czi.get_code(0, 'H') == 1
    assert czi.get_code(0, 'X') == 512
    assert czi.get_code(0, 'Y') == 512


@pytest.mark.skipif(not DATA.exists(), reason='no cached data')
def test_nscenes(czi):
    assert czi.n_scenes == 2


@pytest.mark.skipif(not DATA.exists(), reason='no cached data')
def test_channel_names(czi):
    assert czi.get_channel_names(0) == ['AF488-T1', 'AF405-T2', 'AF555-T2']


@pytest.mark.skipif(not DATA.exists(), reason='no cached data')
@patch('matplotlib.pyplot.show')
def test_view(arg, czi):
    arr = czi.view()
    plt.imshow(arr, cmap='gray')
    plt.show()
