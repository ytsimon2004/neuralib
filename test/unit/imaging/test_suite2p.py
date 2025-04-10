from pathlib import Path
from unittest.mock import patch

import pytest

from neuralib.imaging.suite2p import Suite2PResult
from neuralib.io.dataset import load_example_suite2p
from neuralib.plot import plot_figure

DATA_EXISTS = (Path().home() / '.cache' / 'neuralib' / 'tmp' / 's2p').exists()


@pytest.fixture()
def s2p() -> Suite2PResult:
    return load_example_suite2p(quiet=False, cached=True, rename_folder='s2p')


@pytest.mark.skipif(not DATA_EXISTS, reason='no cached data')
def test_getattr(s2p: Suite2PResult):
    assert s2p.f_raw.size == s2p.f_neu.size == s2p.spks.size


@pytest.mark.skipif(not DATA_EXISTS, reason='no cached data')
def test_get_dff(s2p: Suite2PResult):
    from neuralib.imaging.suite2p import get_neuron_signal
    get_neuron_signal(s2p, n=0, dff=True)


@patch('matplotlib.pyplot.show')
@pytest.mark.skipif(not DATA_EXISTS, reason='no cached data')
def test_plot_soma_center(mock_show, s2p: Suite2PResult):
    from neuralib.imaging.suite2p import plot_soma_center

    with plot_figure(None) as ax:
        plot_soma_center(ax, s2p)
