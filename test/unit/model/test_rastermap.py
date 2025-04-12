from tempfile import NamedTemporaryFile

import attrs

from neuralib.io.dataset import load_example_rastermap_2p_result, load_example_suite2p_result
from neuralib.model.rastermap import RasterMapResult, save_rastermap, read_rastermap
from neuralib.model.rastermap.run import run_rastermap


def test_io():
    # load
    ret = load_example_rastermap_2p_result(cached=True, rename_folder='s2p')
    keys = [field.name for field in attrs.fields(RasterMapResult) if field.init]
    for k in ret.asdict():
        assert k in keys

    # save
    with NamedTemporaryFile(suffix='.npy') as file:
        save_rastermap(ret, file.name)
        ret2 = read_rastermap(file.name)
        for k in ret2.asdict():
            assert k in keys


def test_run():
    s2p = load_example_suite2p_result(cached=True, rename_folder='s2p')
    x = run_rastermap(s2p.spks, bin_size=20)
    print(x.asdict())
