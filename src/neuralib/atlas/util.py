from pathlib import Path
from typing import Literal, NamedTuple, Iterable

import numpy as np
import pandas as pd
import polars as pl

from neuralib.atlas.map import NUM_MERGE_LAYER
from neuralib.atlas.typing import Source, HEMISPHERE_TYPE
from neuralib.typing import DataFrame

__all__ = [
    'ALLEN_CCF_10um_BREGMA',
    #
    'SourceCoordinates',
    'iter_source_coordinates',
    'get_margin_merge_level',
    'allen_to_brainrender_coord',
    'as_coords_array',
]

ALLEN_CCF_10um_BREGMA = np.array([540, 0, 570])  # AP, DV, LR
"""allen CCF 10um volume coordinates, refer to allenCCF/Browsing Functions/allenCCFbregma.m"""


class SourceCoordinates(NamedTuple):
    source: Source
    coordinates: np.ndarray
    """AP, DV, ML coordinates. `Array[float, [N, 3]]`"""

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
        area: list[str] | str | None = None,
        source: list[Source] | Source | None = None,
        region_col: str | None = None,
        hemisphere: HEMISPHERE_TYPE = 'both',
        to_brainrender: bool = True,
        source_order: tuple[Source, ...] | None = None,
        inverse_hemisphere: bool = False,
) -> Iterable[SourceCoordinates]:
    """Load allen ccf roi output (merged different color channels).

    :param file: parsed csv file after
    :param area: only show rois in region(s)
    :param source: only show rois from source(s)
    :param region_col: if None, auto infer, and check the lowest merge level contain all the regions specified
    :param hemisphere: which brain hemisphere
    :param to_brainrender: convert the coordinates to brain render
    :param source_order: whether specify the source generator order
    :param inverse_hemisphere: if True, then inverse ML coordinates. default is False.
    :return: Iterable of :class:`SourceCoordinates`
    """
    df = pl.read_csv(file)
    #
    if area is not None and len(area) != 0:
        if isinstance(area, str):
            area = [area]

        if region_col is None:
            region_col = get_margin_merge_level(df, area, 'lowest')

        df = df.filter(pl.col(region_col).is_in(area))
        if df.is_empty():
            raise RuntimeError('check lowest merge level')

    #
    if source is not None and len(source) != 0:
        if isinstance(source, str):
            source = [source]

        df = df.filter(pl.col('source').is_in(source))
        if df.is_empty():
            raise RuntimeError(f'empty df: likely incorrect source selected {source}')

    #
    if hemisphere != 'both':
        df = df.filter(pl.col('hemisphere') == hemisphere)

    #
    coords = as_coords_array(df)

    if inverse_hemisphere:
        coords[:, 2] *= -1

    if to_brainrender:
        coords = allen_to_brainrender_coord(coords)

    if source_order is None:
        source_order = df['source'].unique()

    for src in source_order:
        mx = df.select(pl.col('source') == src).to_numpy()[:, 0]
        yield SourceCoordinates(src, coords[mx])


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

    eval_merge = [f'tree_{i}' for i in range(NUM_MERGE_LAYER)]

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


def allen_to_brainrender_coord(data: DataFrame | np.ndarray) -> np.ndarray:
    """Convert coordinates space of ``AllenCCF`` to ``Brainrender`` coordinates

    :param data: Dataframe with 'AP_location', 'DV_location', 'ML_location' headers.
            Or numpy array with `Array[float, [N, 3]]` or `Array[float, 3]`
    :return: brainrender coordinates. `Array[float, [N, 3]]` with AP, DV, ML coordinates
    """
    coords = as_coords_array(data)

    coords *= 1000
    coords[:, 0] /= -1  # increment toward posterior
    coords[:, 2] /= -1  # increment toward left hemisphere
    bregma = ALLEN_CCF_10um_BREGMA * 10  # pixel to um
    coords += bregma  # roi relative to bregma

    return coords


def as_coords_array(data: DataFrame | np.ndarray) -> np.ndarray:
    """
    Convert dataframe/1D numpy array to coordinates numpy array

    :param data: Dataframe with 'AP_location', 'DV_location', 'ML_location' headers.
        Or numpy array with `Array[float, [N, 3]]` or `Array[float, 3]`
    :return: `Array[float, [N, 3]]` with AP, DV, ML coordinates
    """
    match data:
        case pd.DataFrame():
            coords = pl.from_pandas(data).select('AP_location', 'DV_location', 'ML_location').to_numpy()
        case pl.DataFrame():
            coords = data.select('AP_location', 'DV_location', 'ML_location').to_numpy()
        case np.ndarray():
            if data.ndim == 1:
                data = np.expand_dims(data, axis=0)
            if data.ndim != 2 or data.shape[1] != 3:
                raise ValueError(f'{data.ndim=}, {data.shape=}')
            coords = data
        case _:
            raise TypeError('')

    return coords
