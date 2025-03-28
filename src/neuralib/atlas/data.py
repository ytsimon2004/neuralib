from pathlib import Path
from typing import Literal

import nrrd
import numpy as np
import polars as pl
from brainglobe_atlasapi import BrainGlobeAtlas

from neuralib.io.core import ALLEN_SDK_DIRECTORY, ATLAS_CACHE_DIRECTORY
from neuralib.typing import PathLike
from neuralib.util.deprecation import deprecated_func
from neuralib.util.tqdm import download_with_tqdm
from neuralib.util.verbose import fprint, print_save

__all__ = [
    #
    'get_dorsal_cortex',
    'load_bg_structure_tree',
    #
    'load_allensdk_annotation',
    'load_ccf_annotation',
    'load_ccf_template',
    'load_structure_tree',

]


def load_bg_structure_tree(atlas_name: str = 'allen_mouse_10um',
                           check_latest: bool = True,
                           parse: bool = False) -> pl.DataFrame:
    """
    Load structure dataframe or dict from `brainglobe_atlasapi`

    :param atlas_name: allen source name
    :param check_latest: if check the brainglobe api latest version
    :param parse: whether parse the child and parent in the same row
    :return:
    """
    file = BrainGlobeAtlas(atlas_name, check_latest=check_latest).root_dir / 'structures.csv'
    df = pl.read_csv(file)

    if parse:
        name = df.select(pl.col('acronym').alias('names'), pl.col('id'), pl.col('parent_structure_id').cast(int))
        join_df = name.join(name, left_on='parent_structure_id', right_on='id')
        parent_child = join_df.select(pl.col('names'), pl.col('names_right').alias('parents'))

        return parent_child
    else:
        return df


def get_dorsal_cortex(output_dir: Path | None = None) -> Path:
    """
    Get example dorsal projection annotation svg file

    .. seealso::

        https://community.brain-map.org/t/aligning-dorsal-projection-of-mouse-common-coordinate-framework-with-wide-field-images-of-mouse-brain/140/2

    :param output_dir: Output directory for caching
    :return: Output file path
    """

    if output_dir is None:
        output_dir = ATLAS_CACHE_DIRECTORY

    filename = 'cortical_map_top_down.svg'
    output = output_dir / filename

    if not output.exists():
        url = 'http://connectivity.brain-map.org/assets/cortical_map_top_down.svg'
        content = download_with_tqdm(url)

        with open(output, 'wb') as f:
            f.write(content.getvalue())
            print_save(output, verb='DOWNLOAD')

    return output


# ================ #
# TO BE DEPRECATED #
# ================ #

@deprecated_func(removal_version='0.5.0', remarks='switch brainglobe api instead')
def load_ccf_annotation(output_dir: PathLike | None = None) -> np.ndarray:
    from ._deprecate import _load_ccf_annotation
    return _load_ccf_annotation(output_dir)


@deprecated_func(removal_version='0.5.0', remarks='switch brainglobe api instead')
def load_ccf_template(output_dir: PathLike | None = None) -> np.ndarray:
    from ._deprecate import _load_ccf_template
    return _load_ccf_template(output_dir)


@deprecated_func(removal_version='0.5.0', remarks='switch brainglobe api instead')
def load_structure_tree(version: Literal['2017', 'old'] = '2017', output_dir: PathLike | None = None) -> pl.DataFrame:
    from ._deprecate import _load_structure_tree
    return _load_structure_tree(version, output_dir)


@deprecated_func(removal_version='0.5.0', remarks='switch brainglobe api instead, and probably deprecate allensdk dependency')
def load_allensdk_annotation(resolution: int = 10, output_dir: PathLike | None = None) -> np.ndarray:
    """
    Data Source directly from Allen Institute

    .. seealso::

        https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/

    :param resolution: volume resolution in um. default is 10 um
    :param output_dir: output directory for caching
    :return: Array[uint32, [AP, DV, ML]]
    """
    if output_dir is None:
        output_dir = ALLEN_SDK_DIRECTORY
        if not ALLEN_SDK_DIRECTORY.exists():
            ALLEN_SDK_DIRECTORY.mkdir(exist_ok=True, parents=True)

    file = output_dir / f'annotation_{resolution}.nrrd'

    if not file.exists():
        try:
            from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
        except ImportError as e:
            fprint('Build error from project.toml. Please manually install using "pip install allensdk --no-deps"', vtype='error')
            raise e

        mcapi = MouseConnectivityApi()
        version = MouseConnectivityApi.CCF_VERSION_DEFAULT

        mcapi.download_annotation_volume(version, resolution, file)

    return nrrd.read(file)[0]
