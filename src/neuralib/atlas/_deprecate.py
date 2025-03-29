import dataclasses
import io
import warnings
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import polars as pl
from typing_extensions import Self

from neuralib.atlas.data import load_bg_structure_tree
from neuralib.io.core import CCF_CACHE_DIRECTORY, ATLAS_CACHE_DIRECTORY
from neuralib.typing import PathLike
from neuralib.util.tqdm import download_with_tqdm
from neuralib.util.verbose import fprint, print_save


def _cache_ndarray(url: str, file: PathLike) -> None:
    fprint(f'DOWNLOADING... {file.name} from {url}', vtype='io')

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        content = download_with_tqdm(url)
        data = np.load(content, allow_pickle=True)
        np.save(file, data)


def _load_ccf_annotation(output_dir: PathLike | None = None) -> np.ndarray:
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


def _load_ccf_template(output_dir: PathLike | None = None) -> np.ndarray:
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


def _load_structure_tree(version: Literal['2017', 'old'] = '2017',
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


# ========= #


@dataclasses.dataclass
class _CellAtlas:
    dataframe: pl.DataFrame
    """
    Example::

        ┌────────────────────────────────┬────────────────┬───────────┐
        │ Brain region                   ┆ Volumes [mm^3] ┆ n_neurons │
        │ ---                            ┆ ---            ┆ ---       │
        │ str                            ┆ f64            ┆ i64       │
        ╞════════════════════════════════╪════════════════╪═══════════╡
        │ Abducens nucleus               ┆ 0.015281       ┆ 1324      │
        │ Accessory facial motor nucleus ┆ 0.013453       ┆ 497       │
        │ Accessory olfactory bulb       ┆ 0.6880625      ┆ 189608    │
        │ …                              ┆ …              ┆ …         │
        │ Zona incerta                   ┆ 2.157641       ┆ 136765    │
        │ posteromedial visual area      ┆ 1.2225625      ┆ 197643    │
        └────────────────────────────────┴────────────────┴───────────┘
    """

    @classmethod
    def load_from_csv(cls, file: Path | None = None,
                      ignore_cell_types_info: bool = True,
                      ignore_detail_info: bool = True) -> Self:
        """
        Load/Download the csv file

        :param file: filepath. If None, download from source paper
        :param ignore_cell_types_info: ignore cell types information, only select neuron and volume foreach areas
        :param ignore_detail_info: ignore information in brain subregion
        :return:
        """
        if file is None:
            d = ATLAS_CACHE_DIRECTORY

            if not d.exists():
                d.mkdir(exist_ok=True, parents=True)

            file = d / 'cellatlas.csv'
            if not file.exists():
                cls._request(file)

        df = pl.read_csv(file)

        if ignore_cell_types_info:
            df = df.select('Brain region', 'Neuron [mm^-3]', 'Volumes [mm^3]')

        # total neurons
        df = (
            df.with_columns((pl.col('Neuron [mm^-3]') * pl.col('Volumes [mm^3]')).alias('n_neurons').cast(pl.Int64))
            .drop('Neuron [mm^-3]')
        )

        if ignore_detail_info:
            patterns = (',', '/', r'\(')
            for pt in patterns:
                df = df.filter(~(pl.col('Brain region').str.contains(pt)))

        return cls(df.sort('Brain region'))

    @classmethod
    def _request(cls, output: Path) -> pl.DataFrame:
        """download from paper source"""
        import requests

        url = 'https://journals.plos.org/ploscompbiol/article/file?type=supplementary&id=10.1371/journal.pcbi.1010739.s011'
        resp = requests.get(url)

        if resp.status_code == 200:
            df = pd.read_excel(io.BytesIO(resp.content), sheet_name='Densities BBCAv1')
            pl.from_pandas(df).write_csv(output)
            fprint(f'Download successfully cellatlas csv and save in {output}!', vtype='io')
        else:
            raise RuntimeError('download cellatlas FAIL')

        return df

    @classmethod
    def load_sync_allen_structure_tree(cls, force_save: bool = True) -> pl.DataFrame:
        """
        TODO `ProS` not found
        Based on cellatlas dataframe, create a sync used `acronym` header in allen struct_tree (sorted by name)

        **fields**

        ``name``: `Brain region` found in cellatlas

        ``n_neurons``: `n_neurons` from cellatlas dataframe

        ``acronym``: `acronym` found in the structure_tree csv

        :param force_save: create sync file every time in root directory
        :return: sync_dataframe
            example::

                ┌────────────────────────────────┬────────────────┬───────────┬─────────┐
                │ name                           ┆ Volumes [mm^3] ┆ n_neurons ┆ acronym │
                │ ---                            ┆ ---            ┆ ---       ┆ ---     │
                │ str                            ┆ f64            ┆ i64       ┆ str     │
                ╞════════════════════════════════╪════════════════╪═══════════╪═════════╡
                │ Abducens nucleus               ┆ 0.015281       ┆ 1324      ┆ VI      │
                │ Agranular insular area         ┆ 4.901734       ┆ 242362    ┆ AI      │
                │ …                              ┆ …              ┆ …         ┆ …       │
                │ Visual areas                   ┆ 12.957203      ┆ 1297194   ┆ VIS     │
                │ Zona incerta                   ┆ 2.157641       ┆ 136765    ┆ ZI      │
                │ posteromedial visual area      ┆ 1.2225625      ┆ 197643    ┆ VISpm   │
                └────────────────────────────────┴────────────────┴───────────┴─────────┘
        """
        out = ATLAS_CACHE_DIRECTORY / 'cellatlas_allen_sync.csv'

        if not out.exists() or force_save:
            ctlas = cls.load_from_csv()

            allen = load_bg_structure_tree().select('name', 'acronym').sort('name')

            df = (
                ctlas.dataframe
                .rename({'Brain region': 'name'})
                .join(allen, on='name')
            )

            df.write_csv(out)
            fprint(f'SAVE cellatlas sync csv to {out}', vtype='io')

        return pl.read_csv(out)
