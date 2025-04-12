from unittest.mock import patch

import numpy as np
import pytest
from tifffile import tifffile

from neuralib.imaging.widefield import plot_retinotopic_maps, compute_singular_vector
from neuralib.io.dataset import load_example_retinotopic_data


@pytest.fixture(scope='module')
def sequence() -> np.ndarray:
    seq_path = load_example_retinotopic_data()
    return tifffile.imread(seq_path)


@patch('matplotlib.pyplot.show')
def test_plot_map(mock, sequence: np.ndarray):
    plot_retinotopic_maps(sequence)


def test_compute_svd(sequence: np.ndarray):
    compute_singular_vector(sequence)
