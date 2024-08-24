import numpy as np
from matplotlib import pyplot as plt

from neuralib.calimg.spikes.cascade import cascade_predict
from neuralib.util.tqdm import download_with_tqdm

NEURON_ID = 0


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


def test_oasis_spike():
    pass


def test_all():
    pass


if __name__ == '__main__':
    test_cascade_spike()
