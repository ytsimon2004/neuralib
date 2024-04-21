from __future__ import annotations

import itertools

import numpy as np
from bokeh.palettes import Category20c
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

__all__ = [
    'ColorMapper',
    'get_customized_cmap',
    'ax_colorbar',
    'insert_colorbar'
]


class ColorMapper:  # TODO named as DiscreteColorMapper
    """map color to iterable object

    **Example of a `Dict` palette**

        >>> cmapper = ColorMapper(Category20c, 20)
        >>> x = ['1', '2', '3']
        >>> color_list = [cmapper[i] for i in x]

    **Example of a mpl `str` palette**

        >>> cmapper = ColorMapper('viridis', 20)
        >>> regions = ['rsc', 'vis', 'hpc']
        >>> color_list = [cmapper[r] for r in regions]
    """

    def __init__(self,
                 palette: dict[int, tuple] | str,
                 number: int):
        if isinstance(palette, dict):
            self._color_pool = iter(itertools.cycle(palette[number]))
        elif isinstance(palette, str):  # matplotlib
            import matplotlib as mpl
            mapper = mpl.colormaps[palette].resampled(number)
            self._color_pool = iter(itertools.cycle(mapper(range(number))))
        else:
            raise TypeError('')

        self._key_pool = {}

    def __getitem__(self, item: str):
        if item in self._key_pool:
            return self._key_pool[item]
        else:
            color = next(self._color_pool)
            self._key_pool[item] = color
            return color


def get_customized_cmap(name: str, value: tuple[float, float], numbers: int) -> np.ndarray:
    """
    Generate gradient color map array

    :param name: name of cmap
    :param value: value range, could be 0-1
    :param numbers: number of color (N)
    :return: (N, 4) RGBA
    """
    cmap = plt.get_cmap(name)
    return cmap(np.linspace(*value, numbers))


def ax_colorbar(ax: Axes, height="25%") -> Axes:
    return inset_axes(
        ax,
        width="5%",
        height=height,
        loc='upper left',
        bbox_to_anchor=(1.01, 0., 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0,
    )


def insert_colorbar(ax: Axes, im, **kwargs) -> ColorbarBase:
    cax = ax_colorbar(ax)

    return ax.figure.colorbar(im, cax=cax, **kwargs)
