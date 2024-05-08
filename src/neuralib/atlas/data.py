from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal

import nrrd
import numpy as np
import pandas as pd
import polars as pl

from neuralib.util.io import CCF_CACHE_DIRECTORY, ALLEN_SDK_DIRECTORY
from neuralib.util.tqdm import download_with_tqdm
from neuralib.util.util_type import PathLike
from neuralib.util.util_verbose import fprint

__all__ = [
    'DATA_SOURCE_TYPE',
    #
    'load_ccf_annotation',
    'load_ccf_template',
    'load_structure_tree',
    #
    'load_allensdk_annotation',

]

DATA_SOURCE_TYPE = Literal['ccf_annotation', 'ccf_template', 'allensdk_annotation']


# ===================== #
# AllenCCF Data Source #
# ===================== #

def _cache_nparray(url: str, file: PathLike):
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
    :return:
    """
    if output_dir is None:
        output_dir = CCF_CACHE_DIRECTORY
        if not CCF_CACHE_DIRECTORY.exists():
            CCF_CACHE_DIRECTORY.mkdir(exist_ok=True, parents=True)

    file = output_dir / 'annotation_volume_10um_by_index.npy'

    if not file.exists():
        url = 'https://figshare.com/ndownloader/files/44925493'
        _cache_nparray(url, file)

    return np.load(file, allow_pickle=True)


def load_ccf_template(output_dir: PathLike | None = None) -> np.ndarray:
    """
    Template volume file from AllenCCF pipeline

    .. seealso::

        https://github.com/cortex-lab/allenCCF

    :param output_dir: output directory for caching
    :return:
    """
    if output_dir is None:
        output_dir = CCF_CACHE_DIRECTORY
        if not CCF_CACHE_DIRECTORY.exists():
            CCF_CACHE_DIRECTORY.mkdir(exist_ok=True, parents=True)

    file = output_dir / 'template_volume_10um.npy'

    if not file.exists():
        url = 'https://figshare.com/ndownloader/files/44925496'
        _cache_nparray(url, file)

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
    file = output_dir / filename
    if not file.exists():
        url = f'https://raw.githubusercontent.com/cortex-lab/allenCCF/master/{filename}'
        content = download_with_tqdm(url)
        df = pd.read_csv(content)
        df.to_csv(file)
        fprint(f'DOWNLOAD! {filename} in {output_dir}', vtype='io')

    ret = (pl.read_csv(file)
           .with_columns(pl.col('parent_structure_id').fill_null(-1))
           .with_columns(pl.col('structure_id_path').map_elements(lambda it: tuple(map(int, it[1:-1].split('/'))))))
    #
    if '2017' in Path(file).name:
        ret = ret.with_columns(pl.col('atlas_id').fill_null(-2))
    else:
        ret = ret.with_columns(pl.col('name').alias('safe_name'))

    return ret


# ===================== #
# Allen SDK Data Source #
# ===================== #

def load_allensdk_annotation(resolution: int = 10,
                             output_dir: PathLike | None = None) -> np.ndarray:
    """
    Data Source directly from Allen Institute

    .. seealso::

        https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/

    :param resolution: volume resolution in um. default is 10 um
    :param output_dir: output directory for caching
    :return:
    """
    if output_dir is None:
        output_dir = ALLEN_SDK_DIRECTORY
        if not ALLEN_SDK_DIRECTORY.exists():
            ALLEN_SDK_DIRECTORY.mkdir(exist_ok=True, parents=True)

    file = output_dir / f'annotation_{resolution}.nrrd'

    if not file.exists():
        from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
        mcapi = MouseConnectivityApi()
        version = MouseConnectivityApi.CCF_VERSION_DEFAULT

        mcapi.download_annotation_volume(version, resolution, file)

    return nrrd.read(file)[0]
