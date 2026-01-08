from tempfile import NamedTemporaryFile
from unittest.mock import patch

import attrs
import pytest

from neuralib.imglib.io import load_sequence
from neuralib.io.dataset import load_example_rastermap_2p_result, load_example_rastermap_2p_cache
from neuralib.model.rastermap import RasterMapResult, save_rastermap, read_rastermap
from neuralib.model.rastermap.plot import plot_rastermap, plot_cellular_spatial, plot_wfield_spatial, Covariant
from neuralib.model.rastermap.run import run_rastermap


def test_io():
    # load
    ret = load_example_rastermap_2p_result(cached=True, rename_file='rastermap.npy')
    keys = [field.name for field in attrs.fields(RasterMapResult) if field.init]
    for k in ret.asdict():
        assert k in keys

    # save
    with NamedTemporaryFile(suffix='.npy') as file:
        save_rastermap(ret, file.name)
        ret2 = read_rastermap(file.name)
        for k in ret2.asdict():
            assert k in keys


@pytest.mark.skip(reason="test locally")
@patch('matplotlib.pyplot.show')
def test_run_plot_2p(mock):
    cache = load_example_rastermap_2p_cache(cached=True, rename_file='raster_2p')
    t = cache['image_time']
    xy = cache['xy_pos']

    ret = run_rastermap(cache['neural_activity'], bin_size=50)
    pos = Covariant('position', dtype='continuous', time=t, value=cache['position'])
    vel = Covariant('velocity', dtype='continuous', time=t, value=cache['velocity'])
    pupil = Covariant('pupil', dtype='continuous', time=t, value=cache['pupil_area'])
    plot_rastermap(ret, t, time_range=(0, 300), covars=[pos, vel, pupil])
    plot_cellular_spatial(ret, xpos=xy[0], ypos=xy[1])


@pytest.mark.skip(reason="memory heavy computing")
def test_run_wfield():
    d = ...
    x = load_sequence(d)
    ret = run_rastermap(x, bin_size=500, dtype='wfield')
    act = ...
    plot_rastermap(ret, act)
    plot_wfield_spatial(ret, x.shape[0], x.shape[1])
