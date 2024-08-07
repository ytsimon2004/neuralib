from __future__ import annotations

import itertools

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from neuralib.util.deprecation import deprecated_class

__all__ = [
    'ColorMapper',
    'DiscreteColorMapper',
    'get_customized_cmap',
    'ax_colorbar',
    'insert_colorbar'
]


class DiscreteColorMapper:
    """map color to iterable object

    **Example of a ``dict`` palette**

        >>> palette = {3: ('#3182bd', '#6baed6', '#9ecae1'),
        ...            5: ('#a1d99b', '#c7e9c0', '#756bb1', '#9e9ac8', '#bcbddc')}
        >>> cmapper = DiscreteColorMapper(palette, 5)
        >>> x = ['1', '2', '3']
        >>> color_list = [cmapper[i] for i in x]

    **Example of a mpl ``str`` palette**

        >>> cmapper = DiscreteColorMapper('viridis', 20)
        >>> regions = ['rsc', 'vis', 'hpc']
        >>> color_list = [cmapper[r] for r in regions]
    """

    def __init__(self,
                 palette: dict[int, tuple[str, ...]] | str,
                 number: int):
        """

        :param palette: A custom dictionary color palette or a Matplotlib colormap name
        :param number: Number of colors to cycle through in the palette or colormap
        """
        if isinstance(palette, dict):
            if number not in palette:
                raise ValueError(f"Palette dictionary does not contain key '{number}'.")
            self._color_pool = iter(itertools.cycle(palette[number]))
        elif isinstance(palette, str):
            import matplotlib as mpl
            mapper = mpl.colormaps[palette].resampled(number)
            self._color_pool = iter(itertools.cycle(mapper(range(number))))
        else:
            raise TypeError('palette must be a dict or str')

        self._key_pool = {}

    def __getitem__(self, item: str) -> str | np.ndarray:
        """get single color"""
        if item in self._key_pool:
            return self._key_pool[item]
        else:
            color = next(self._color_pool)
            self._key_pool[item] = color
            return color


@deprecated_class(new_class='DiscreteColorMapper')
class ColorMapper(DiscreteColorMapper):
    """Deprecated: Use DiscreteColorMapper instead."""
    pass


def get_customized_cmap(name: str, value: tuple[float, float], numbers: int) -> np.ndarray:
    """
    Generate gradient color map array.
    `N` = number of color

    :param name: name of cmap
    :param value: value range, could be 0-1
    :param numbers: `N`
    :return: RGBA. `Array[float, [N, 4]]`
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


if __name__ == '__main__':
    ColorMapper('viridis', 20)
