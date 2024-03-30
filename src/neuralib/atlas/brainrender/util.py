from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl
from brainrender._colors import get_random_colors

from neuralib.util.util_type import DataFrame

__all__ = ['get_color', 'roi_points_converter']


def get_color(i, color_pattern: str | tuple[str, ...] | list[str]) -> str:
    """
    get color
    :param i: idx of the color pattern list
    :param color_pattern: color pattern list or single element str
    :return:
        color name
    """
    if isinstance(color_pattern, (list, tuple)):
        if i >= len(color_pattern):
            color_pattern.extend(get_random_colors(i + 2 - len(color_pattern)))
        return color_pattern[i]
    elif isinstance(color_pattern, str):
        return color_pattern


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
    from neuralib.atlas.util import ALLEN_CCF_10um_BREGMA

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
