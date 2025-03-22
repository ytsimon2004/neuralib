import warnings
from pathlib import Path
from typing import Literal

import nrrd
import numpy as np
import pandas as pd
import polars as pl
from brainglobe_atlasapi import BrainGlobeAtlas

from neuralib.io.core import CCF_CACHE_DIRECTORY, ALLEN_SDK_DIRECTORY, ATLAS_CACHE_DIRECTORY
from neuralib.typing import PathLike
from neuralib.util.deprecation import deprecated_func
from neuralib.util.tqdm import download_with_tqdm
from neuralib.util.verbose import fprint, print_save

__all__ = [
    'DATA_SOURCE_TYPE',
    #
    'load_ccf_annotation',
    'load_ccf_template',
    'load_structure_tree',
    #
    'load_allensdk_annotation',
    #
    'load_bg_structure_tree'

]

DATA_SOURCE_TYPE = Literal['ccf_annotation', 'ccf_template', 'allensdk_annotation']


# ===================== #
# AllenCCF Data Source #
# ===================== #

def _cache_ndarray(url: str, file: PathLike) -> None:
    fprint(f'DOWNLOADING... {file.name} from {url}', vtype='io')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        content = download_with_tqdm(url)
        data = np.load(content, allow_pickle=True)
        np.save(file, data)


def load_ccf_annotation(output_dir: PathLike | None = None) -> np.ndarray:
    """
    Annotation volume file from AllenCCF pipeline

    .. seealso::

        https://github.com/cortex-lab/allenCCF

    :param output_dir: output directory for caching
    :return: Array[uint16, [AP, DV, ML]]
    """
    if output_dir is None:
        output_dir = CCF_CACHE_DIRECTORY
        if not CCF_CACHE_DIRECTORY.exists():
            CCF_CACHE_DIRECTORY.mkdir(exist_ok=True, parents=True)

    file = output_dir / 'annotation_volume_10um_by_index.npy'

    if not file.exists():
        url = 'https://figshare.com/ndownloader/files/44925493'
        _cache_ndarray(url, file)

    return np.load(file, allow_pickle=True)


def load_ccf_template(output_dir: PathLike | None = None) -> np.ndarray:
    """
    Template volume file from AllenCCF pipeline

    .. seealso::

        https://github.com/cortex-lab/allenCCF

    :param output_dir: output directory for caching
    :return: Array[uint16, [AP, DV, ML]]
    """
    if output_dir is None:
        output_dir = CCF_CACHE_DIRECTORY
        if not CCF_CACHE_DIRECTORY.exists():
            CCF_CACHE_DIRECTORY.mkdir(exist_ok=True, parents=True)

    file = output_dir / 'template_volume_10um.npy'

    if not file.exists():
        url = 'https://figshare.com/ndownloader/files/44925496'
        _cache_ndarray(url, file)

    return np.load(file, allow_pickle=True)


def load_structure_tree(version: Literal['2017', 'old'] = '2017',
                        output_dir: PathLike | None = None) -> pl.DataFrame:
    """
    Load structure tree dataframe

    :param version: {'2017', 'old'}
    :param output_dir: output directory for caching
    :return: structure tree dataframe

    ::

        ┌───────────┬──────────┬───────────────────────────────┬─────────┬──────────┬─────────────┬───────────────┬────────┬─────────────────────┬───────┬──────────┬─────────────┬───────────────────┬───────────────────┬─────────────────────────┬──────────────────────────────┬────────┬───────────┬──────────────────────┬──────────────┬───────────────────────────────┐
        │ id        ┆ atlas_id ┆ name                          ┆ acronym ┆ st_level ┆ ontology_id ┆ hemisphere_id ┆ weight ┆ parent_structure_id ┆ depth ┆ graph_id ┆ graph_order ┆ structure_id_path ┆ color_hex_triplet ┆ neuro_name_structure_id ┆ neuro_name_structure_id_path ┆ failed ┆ sphinx_id ┆ structure_name_facet ┆ failed_facet ┆ safe_name                     │
        │ ---       ┆ ---      ┆ ---                           ┆ ---     ┆ ---      ┆ ---         ┆ ---           ┆ ---    ┆ ---                 ┆ ---   ┆ ---      ┆ ---         ┆ ---               ┆ ---               ┆ ---                     ┆ ---                          ┆ ---    ┆ ---       ┆ ---                  ┆ ---          ┆ ---                           │
        │ i64       ┆ f64      ┆ str                           ┆ str     ┆ str      ┆ i64         ┆ i64           ┆ i64    ┆ f64                 ┆ i64   ┆ i64      ┆ i64         ┆ list[i64]         ┆ str               ┆ str                     ┆ str                          ┆ str    ┆ i64       ┆ i64                  ┆ i64          ┆ str                           │
        ╞═══════════╪══════════╪═══════════════════════════════╪═════════╪══════════╪═════════════╪═══════════════╪════════╪═════════════════════╪═══════╪══════════╪═════════════╪═══════════════════╪═══════════════════╪═════════════════════════╪══════════════════════════════╪════════╪═══════════╪══════════════════════╪══════════════╪═══════════════════════════════╡
        │ 997       ┆ -1.0     ┆ root                          ┆ root    ┆ null     ┆ 1           ┆ 3             ┆ 8690   ┆ -1.0                ┆ 0     ┆ 1        ┆ 0           ┆ [997]             ┆ FFFFFF            ┆ null                    ┆ null                         ┆ f      ┆ 1         ┆ 385153371            ┆ 734881840    ┆ root                          │
        │ 8         ┆ 0.0      ┆ Basic cell groups and regions ┆ grey    ┆ null     ┆ 1           ┆ 3             ┆ 8690   ┆ 997.0               ┆ 1     ┆ 1        ┆ 1           ┆ [997, 8]          ┆ BFDAE3            ┆ null                    ┆ null                         ┆ f      ┆ 2         ┆ 2244697386           ┆ 734881840    ┆ Basic cell groups and regions │
        │ …         ┆ …        ┆ …                             ┆ …       ┆ …        ┆ …           ┆ …             ┆ …      ┆ …                   ┆ …     ┆ …        ┆ …           ┆ …                 ┆ …                 ┆ …                       ┆ …                            ┆ …      ┆ …         ┆ …                    ┆ …            ┆ …                             │
        │ 65        ┆ 715.0    ┆ parafloccular sulcus          ┆ pfs     ┆ null     ┆ 1           ┆ 3             ┆ 8690   ┆ 1040.0              ┆ 3     ┆ 1        ┆ 1324        ┆ [997, 1024, … 65] ┆ AAAAAA            ┆ null                    ┆ null                         ┆ f      ┆ 1325      ┆ 771629690            ┆ 734881840    ┆ parafloccular sulcus          │
        │ 624       ┆ 926.0    ┆ Interpeduncular fossa         ┆ IPF     ┆ null     ┆ 1           ┆ 3             ┆ 8690   ┆ 1024.0              ┆ 2     ┆ 1        ┆ 1325        ┆ [997, 1024, 624]  ┆ AAAAAA            ┆ null                    ┆ null                         ┆ f      ┆ 1326      ┆ 1476705011           ┆ 734881840    ┆ Interpeduncular fossa         │
        │ 304325711 ┆ -2.0     ┆ retina                        ┆ retina  ┆ null     ┆ 1           ┆ 3             ┆ 8690   ┆ 997.0               ┆ 1     ┆ 1        ┆ 1326        ┆ [997, 304325711]  ┆ 7F2E7E            ┆ null                    ┆ null                         ┆ f      ┆ 1327      ┆ 3295290839           ┆ 734881840    ┆ retina                        │
        └───────────┴──────────┴───────────────────────────────┴─────────┴──────────┴─────────────┴───────────────┴────────┴─────────────────────┴───────┴──────────┴─────────────┴───────────────────┴───────────────────┴─────────────────────────┴──────────────────────────────┴────────┴───────────┴──────────────────────┴──────────────┴───────────────────────────────┘

    """
    #
    if output_dir is None:
        output_dir = CCF_CACHE_DIRECTORY
        if not CCF_CACHE_DIRECTORY.exists():
            CCF_CACHE_DIRECTORY.mkdir(exist_ok=True, parents=True)
    #
    if version == '2017':
        filename = 'structure_tree_safe_2017.csv'
    elif version == 'old':
        filename = 'structure_tree_safe.csv'
    else:
        raise ValueError('')

    #
    output = output_dir / filename
    if not output.exists():
        url = f'https://raw.githubusercontent.com/cortex-lab/allenCCF/master/{filename}'
        content = download_with_tqdm(url)
        df = pd.read_csv(content)
        df.to_csv(output)
        print_save(output, verb='DOWNLOAD')

    ret = (pl.read_csv(output)
           .with_columns(pl.col('parent_structure_id').fill_null(-1))
           .with_columns(pl.col('structure_id_path')
                         .map_elements(lambda it: tuple(map(int, it[1:-1].split('/'))),
                                       return_dtype=pl.List(pl.Int64))))
    #
    if '2017' in Path(output).name:
        ret = ret.with_columns(pl.col('atlas_id').fill_null(-2))
    else:
        ret = ret.with_columns(pl.col('name').alias('safe_name'))

    return ret


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


# ===================== #
# Allen SDK Data Source #
# ===================== #

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


# =========================== #
# BrainGlobeAtlas Data Source #
# =========================== #

def load_bg_structure_tree(source: str = 'allen_mouse_10um',
                           check_latest: bool = True,
                           parse: bool = False) -> pl.DataFrame:
    """
    Load structure dataframe or dict from `brainglobe_atlasapi`

    :param source: allen source name
    :param check_latest: if check the brainglobe api latest version
    :param parse: whether parse the child and parent in the same row
    :return:
    """
    file = BrainGlobeAtlas(source, check_latest=check_latest).root_dir / 'structures.csv'
    df = pl.read_csv(file)

    if parse:
        name = df.select(pl.col('acronym').alias('names'), pl.col('id'), pl.col('parent_structure_id').cast(int))
        join_df = name.join(name, left_on='parent_structure_id', right_on='id')
        parent_child = join_df.select(pl.col('names'), pl.col('names_right').alias('parents'))

        return parent_child
    else:
        return df
