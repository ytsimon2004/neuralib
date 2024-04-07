from __future__ import annotations

import io
from io import BytesIO
from pathlib import Path
from typing import ClassVar, TypedDict, Literal

import allensdk.core.structure_tree
import numpy as np
import pandas as pd
import polars as pl
from allensdk.core.reference_space_cache import ReferenceSpaceCache

from neuralib.atlas.util import PLANE_TYPE, ALLEN_SOURCE_TYPE
from neuralib.atlas.view import AbstractSliceView
from neuralib.util.io import CCF_CACHE_DIRECTORY
from neuralib.util.tqdm import download_with_tqdm
from neuralib.util.util_type import PathLike, DataFrame
from neuralib.util.util_verbose import fprint
from neuralib.util.utils import uglob

__all__ = ['AllenReferenceWrapper']


class StructureTreeDict(TypedDict):
    """allen structure tree dict"""
    acronym: str
    graph_id: int
    graph_order: int
    id: int
    name: str
    structure_id_path: list[int]
    structure_set_ids: list[int]
    rgb_triplet: list[int]


class AllenReferenceWrapper:
    """Class for load the data in allen"""

    REFERENCE_SPACE_KEY: ClassVar[str] = 'ccf_2017'
    STRUCTURE_GRAPH_ID: ClassVar[int] = 1

    def __init__(self,
                 resolution: int = 10,
                 output: PathLike | None = None):
        """
        :param resolution: resolution in um
        :param output: output directory for downloading cache. By default, save in ``CCF_CACHE_DIRECTORY``
        """

        if output is None:
            output = CCF_CACHE_DIRECTORY

        if not output.exists():
            output.mkdir(parents=True, exist_ok=True)

        self.source_root = Path(output)
        self.reference = ReferenceSpaceCache(resolution,
                                             self.REFERENCE_SPACE_KEY,
                                             manifest=self.source_root / 'manifest.json')

    # ============== #
    # Structure Tree #
    # ============== #

    @classmethod
    def load_structure_tree(cls, version: Literal['2017', 'old'] = '2017') -> pl.DataFrame:
        """
        Load structure tree dataframe

        :param version: {'2017', 'old'}
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
        if version == '2017':
            filename = 'structure_tree_safe_2017.csv'
        elif version == 'old':
            filename = 'structure_tree_safe.csv'
        else:
            raise ValueError('')

        #
        filepath = CCF_CACHE_DIRECTORY / filename
        if not filepath.exists():
            cls._request_structure_tree(filepath, filename)

        ret = (pl.read_csv(filepath)
               .with_columns(pl.col('parent_structure_id').fill_null(-1))
               .with_columns(pl.col('structure_id_path').map_elements(lambda it: tuple(map(int, it[1:-1].split('/'))))))
        #
        if '2017' in Path(filepath).name:
            ret = ret.with_columns(pl.col('atlas_id').fill_null(-2))
        else:
            ret = ret.with_columns(pl.col('name').alias('safe_name'))

        return ret

    @classmethod
    def _request_structure_tree(cls, dest: PathLike, filename: str) -> Path:
        url = f'https://raw.githubusercontent.com/cortex-lab/allenCCF/master/{filename}'
        resp = download_with_tqdm(url)

        if resp.status_code == 200:
            df = pd.read_csv(io.BytesIO(resp.content))
            pl.from_pandas(df).write_csv(dest)
            fprint(f'DOWNLOAD! {filename} in {dest.parent}', vtype='io')

        else:
            raise RuntimeError('download structure FAIL!')

        return dest

    def structure_tree(self) -> allensdk.core.structure_tree.StructureTree:
        return self.reference.get_structure_tree(structure_graph_id=self.STRUCTURE_GRAPH_ID)

    def get_structures_by_name(self, name: list[str]) -> list[StructureTreeDict]:
        return self.structure_tree().get_structures_by_name(name)

    # ========== #
    # Annotation #
    # ========== #

    @classmethod
    def load_slice_view(cls,
                        source: ALLEN_SOURCE_TYPE,
                        plane_type: PLANE_TYPE,
                        resolution: int = 10) -> AbstractSliceView:
        """
        Load the slice view

        :param source: ``ALLEN_SOURCE_TYPE``
        :param plane_type: ``PLANE_TYPE``
        :param resolution: resolution in um
        :return: :class:`~neuralib.atlas.view.AbstractSliceView()`
        """

        if source in ('npy', 'nrrd'):
            pattern = f'annotation*{resolution}*.{source}'

            try:
                f = uglob(CCF_CACHE_DIRECTORY, pattern)
            except FileNotFoundError:
                f = cls._request_allen_src(source, resolution, CCF_CACHE_DIRECTORY)

            if source == 'npy':
                data = np.load(f)
            elif source == 'nrrd':
                import nrrd
                data = nrrd.read(f)[0]
            else:
                raise ValueError('')

        elif source == 'template':
            f = CCF_CACHE_DIRECTORY / f'template_volume_{resolution}um.npy'
            if not f.exists():
                f = cls._request_allen_src('template', resolution, CCF_CACHE_DIRECTORY)
            data = np.load(f)

        else:
            raise ValueError('')

        return AbstractSliceView(source, plane_type, resolution, data)

    @classmethod
    def _request_allen_src(cls, src_type: ALLEN_SOURCE_TYPE,
                           resolution: int,
                           dest: PathLike) -> Path:
        """

        :param src_type: ALLEN_SOURCE_TYPE = Literal['npy', 'nrrd', 'template']
                * nrrd: source directly from Alleninstitute
                    Seealso: https://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/
                * annotation / template: adapted version from cortex-lab
                    Seealso: https://github.com/cortex-lab/allenCCF
        :param resolution:
        :param dest:
        :return:
        """

        if src_type == 'nrrd':
            from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
            mcapi = MouseConnectivityApi()
            version = MouseConnectivityApi.CCF_VERSION_DEFAULT
            filename = f'annotation_{resolution}.nrrd'
            mcapi.download_annotation_volume(version, resolution, dest / filename)
            out = Path(dest) / filename
            fprint(f'DOWNLOAD! {filename} in {dest}', vtype='io')

        elif src_type in ('annotation', 'template'):

            if src_type == 'annotation':
                url = 'https://figshare.com/ndownloader/files/44925493'
                filename = 'annotation_volume_10um_by_index.npy'
            elif src_type == 'template':
                url = 'https://figshare.com/ndownloader/files/44925496'
                filename = 'template_volume_10um.npy'
            else:
                raise ValueError('')

            out = Path(dest) / filename
            fprint(f'DOWNLOADING... {filename} from {url}', vtype='io')

            resp = download_with_tqdm(url)
            if resp.status_code == 200:
                fprint(f'DOWNLOAD! {filename} in {dest}', vtype='io')

                with BytesIO(resp.content) as f:

                    data = np.load(f, allow_pickle=True)
                np.save(out, data)

            else:
                raise RuntimeError('download allen src FAIL!')
        else:
            raise ValueError('')

        return out

    # ====== #
    # Others #
    # ====== #

    @classmethod
    def load_structure_voxel(cls, to_pandas=False) -> DataFrame:
        """TODO not test yet"""
        file = CCF_CACHE_DIRECTORY / 'voxel_count_and_differences.csv'
        df = (
            pl.read_csv(file)
            .select(pl.col(['acronym', 'name', 'total_voxel_count']))
            .drop_nulls()
            .sort('total_voxel_count', descending=True)
        )

        if to_pandas:
            df = df.to_pandas()

        return df
