from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import Literal, NamedTuple, Iterable

import numpy as np
import pandas as pd
import polars as pl

from neuralib.atlas.data import load_structure_tree
from neuralib.atlas.map import NUM_MERGE_LAYER
from neuralib.atlas.type import Source, HEMISPHERE_TYPE
from neuralib.util.util_type import DataFrame
from neuralib.util.util_verbose import fprint

__all__ = [
    'ALLEN_CCF_10um_BREGMA',
    'PLANE_TYPE',
    #
    'SourceCoordinates',
    'iter_source_coordinates',
    'get_margin_merge_level',
    'roi_points_converter',
    'create_allen_structure_dict'
]

# allen CCF 10um volume coordinates, refer to allenCCF/Browsing Functions/allenCCFbregma.m
ALLEN_CCF_10um_BREGMA = np.array([540, 0, 570])  # AP, DV, LR

PLANE_TYPE = Literal['coronal', 'sagittal', 'transverse']


class SourceCoordinates(NamedTuple):
    source: Source
    coordinates: np.ndarray
    """(N, 3) with ap, dv, ml"""

    axes_repr: tuple[str, str, str] = ('ap', 'dv', 'ml')

    @property
    def ap(self) -> np.ndarray:
        return self.coordinates[:, 0]

    @property
    def dv(self) -> np.ndarray:
        return self.coordinates[:, 1]

    @property
    def ml(self) -> np.ndarray:
        return self.coordinates[:, 2]


def iter_source_coordinates(
        file: Path,
        *,
        only_areas: list[str] | str | None = None,
        region_col: str | None = None,
        hemisphere: HEMISPHERE_TYPE = 'both',
        to_brainrender: bool = True,
        to_um: bool = True,
        ret_order: tuple[Source, ...] | None = ('pRSC', 'aRSC', 'overlap')
) -> Iterable[SourceCoordinates]:
    """Load allen ccf roi output (merged different color channels).

    :param file: parsed csv file after
    :param only_areas: only show rois in region(s)
    :param region_col: if None, auto infer, and check the lowest merge level contain all the regions specified
    :param hemisphere
    :param to_brainrender: convert the coordinates to brain render
    :param to_um
    :param ret_order: whether specify the source generator order
    :return: :class:`SourceCoordinates`
    """
    df = pl.read_csv(file)
    #
    if only_areas is not None and len(only_areas) != 0:
        if isinstance(only_areas, str):
            only_areas = [only_areas]

        if region_col is None:
            region_col = get_margin_merge_level(df, only_areas, 'lowest')

        df = df.filter(pl.col(region_col).is_in(only_areas))
        fprint(f'using {region_col} for {only_areas}')

        if df.is_empty():
            raise RuntimeError('check lowest merge level')
    #
    if hemisphere != 'both':
        df = df.filter(pl.col('hemi.') == hemisphere)

    #
    points = roi_points_converter(df, to_brainrender=to_brainrender, to_um=to_um)

    if ret_order is None:
        ret_order = df['source'].unique()

    for src in ret_order:
        mask = df.select(pl.col('source') == src).to_numpy()[:, 0]
        yield SourceCoordinates(src, points[mask])


def get_margin_merge_level(df: pl.DataFrame,
                           areas: list[str] | str,
                           margin: Literal['lowest', 'highest']) -> str:
    """Get the lowest or highest merge level (i.e., parsed_csv) containing all the regions

    :param df: parsed csv
    :param areas: an area or a list of areas
    :param margin: get the either lowest of highest merge level for a given area
    :return: col name if parsed csv
    """
    if not isinstance(areas, (tuple, list)):
        areas = [areas]

    eval_merge = [f'merge_ac_{i}' for i in range(NUM_MERGE_LAYER)]

    if margin == 'lowest':
        level = eval_merge
    elif margin == 'highest':
        level = reversed(eval_merge)
    else:
        raise ValueError('')

    for lv in level:
        ls = df[lv]
        if np.all([a in ls for a in areas]):
            return lv

    raise ValueError(f'{areas} not found')


def roi_points_converter(dat: DataFrame | np.ndarray,
                         to_brainrender: bool = True,
                         to_um: bool = True) -> np.ndarray:
    """
    convert coordinates of `allenccf` roi points from parsed dataframe

    :param dat: Dataframe with 'AP_location', 'DV_location', 'ML_location' headers.
            Or numpy array with (N, 3), or (3,)
    :param to_brainrender: coordinates to `brainrender`
    :param to_um:

    :return (N, 3),
        N: number of roi
        3: ap, dv, ml
    """
    if isinstance(dat, pd.DataFrame):
        dat = pl.from_pandas(dat)
        points = dat.select('AP_location', 'DV_location', 'ML_location').to_numpy()
    elif isinstance(dat, pl.DataFrame):
        points = dat.select('AP_location', 'DV_location', 'ML_location').to_numpy()
    elif isinstance(dat, np.ndarray):
        if dat.ndim == 1:
            dat = np.expand_dims(dat, axis=0)

        if dat.ndim == 2 and dat.shape[1] != 3:
            raise ValueError('')

        points = dat
    else:
        raise TypeError('')

    #
    if to_um:
        points *= 1000  # um

    if to_brainrender:
        points[:, 0] /= -1  # increment toward posterior
        points[:, 2] /= -1  # increment toward left hemisphere
        bregma = ALLEN_CCF_10um_BREGMA * 10  # pixel to um
        points += bregma  # roi relative to bregma

    return points


def create_allen_structure_dict(verbose=False) -> dict[str, str]:
    """
    Get the acronym/name pairing from structure_tree.csv

    :return: key: acronym; value: full name
    """
    tree = load_structure_tree()
    tree = tree.select('name', 'acronym').sort('name')

    ret = {
        acry: name
        for name, acry in tree.iter_rows()
    }
    if verbose:
        pprint(ret)

    return ret
