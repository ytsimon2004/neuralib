import pytest

from neuralib.io import NEUROLIB_DATASET_DIRECTORY
from neuralib.io.dataset import load_example_lsm
from neuralib.scan.lsm import lsm_file

DATA = NEUROLIB_DATASET_DIRECTORY / 'test.lsm'
DOWNLOAD_CACHE = False

if not DATA.exists() and DOWNLOAD_CACHE:
    load_example_lsm(cached=True)


@pytest.fixture(scope='module')
def lsm():
    with lsm_file(DATA) as lsm:
        yield lsm


@pytest.mark.skipif(not DATA.exists(), reason='no cached data')
def test_config(lsm):
    assert lsm.file_type == '.lsm'
    assert lsm.get_channel_names() == ['Ch1-T1', 'Ch2-T1', 'Ch1-T2']
    assert lsm.dimcode == 'ZCYX'
    assert lsm.n_scenes == 1
