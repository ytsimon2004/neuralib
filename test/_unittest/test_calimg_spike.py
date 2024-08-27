import unittest
from typing import Callable
from unittest.mock import patch

import numpy as np
from matplotlib import pyplot as plt

from neuralib.calimg.spikes.cascade import cascade_predict
from neuralib.calimg.spikes.oasis import oasis_dcnv
from neuralib.util.tqdm import download_with_tqdm

NEURON_ID = 1


def get_example_dff():
    """download example of Thy1_GCaMP6s_30hz.npy from drive.`Array[float, [n_neurons, n_frames]]`.
    Example has 100 neurons and 5000 frames
    """
    url = 'https://drive.google.com/uc?export=download&id=1ZBZKaoWbizKrbhkjYdTkDWq_ggFc-475'
    content = download_with_tqdm(url)
    return np.load(content, allow_pickle=True)


def test_cascade_spike():
    dff = get_example_dff()
    spks = cascade_predict(dff, model_type='Global_EXC_30Hz_smoothing100ms')
    plt.plot(dff[NEURON_ID, :], label='dff')
    plt.plot(spks[NEURON_ID, :], label='cascade')
    plt.legend()
    plt.show()


def test_oasis_spike_batch():
    dff = get_example_dff()
    f = dff.astype(np.float32)

    spks = oasis_dcnv(f, 1.5, 30)
    plt.plot(dff[NEURON_ID], label='dff')
    plt.plot(spks[NEURON_ID], label='oasis')
    plt.legend()
    plt.show()


def test_oasis_spike_single():
    dff = get_example_dff()[NEURON_ID, :]
    f = dff.astype(np.float32)

    spks = oasis_dcnv(f, 1.5, 30)
    plt.plot(dff, label='dff')
    plt.plot(spks, label='oasis')
    plt.legend()
    plt.show()


class TestPlotting(unittest.TestCase):

    @patch('matplotlib.pyplot.show')
    def plt_close(self, f: Callable, mock_show, *args, **kwargs):
        try:
            f(*args, **kwargs)
            plt.clf()
            plt.close('all')
        except Exception as e:
            self.fail(f'Plotting function raised an exception: {e}')

    def test_plotting_func_runs(self):
        self.plt_close(test_cascade_spike)
        self.plt_close(test_oasis_spike_batch)
        self.plt_close(test_oasis_spike_single)


if __name__ == '__main__':
    unittest.main()
