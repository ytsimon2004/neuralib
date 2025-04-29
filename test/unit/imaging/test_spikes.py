import numpy as np
import pytest

from neuralib.imaging.suite2p import get_neuron_signal
from neuralib.io import NEUROLIB_DATASET_DIRECTORY
from neuralib.io.dataset import load_example_suite2p_result

DATA_EXISTS = (NEUROLIB_DATASET_DIRECTORY / 's2p').exists()


@pytest.fixture(scope='module')
def df_f() -> np.ndarray:
    s2p = load_example_suite2p_result(quiet=False, cached=True, rename_folder='s2p')
    return get_neuron_signal(s2p, n=0)[0][: 10000]


@pytest.mark.skipif(not DATA_EXISTS, reason='no cached data')
def test_oasis(df_f):
    from neuralib.imaging.spikes import oasis_dcnv
    oasis_dcnv(df_f, tau=1.15, fs=30)


@pytest.mark.skipif(not DATA_EXISTS, reason='no cached data')
def test_cascade(df_f):
    from neuralib.imaging.spikes import cascade_predict
    cascade_predict(df_f, model_type='Global_EXC_30Hz_smoothing100ms')
