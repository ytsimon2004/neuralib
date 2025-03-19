import itertools

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import ColorbarBase
from mpl_toolkits.axes_grid1 import make_axes_locatable

__all__ = [
    'DiscreteColorMapper',
    'get_customized_cmap',
    'insert_colorbar',
    'insert_cyclic_colorbar'
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


def get_customized_cmap(name: str,
                        value: tuple[float, float],
                        numbers: int,
                        endpoint: bool = True) -> np.ndarray:
    """
    Generate gradient color map array.
    `N` = number of color

    :param name: name of cmap
    :param value: value range, could be 0-1
    :param numbers: `N`
    :param endpoint: If cyclic colormap, then used `False`
    :return: RGBA. `Array[float, [N, 4]]`
    """
    cmap = plt.get_cmap(name)
    return cmap(np.linspace(*value, numbers, endpoint=endpoint))


def insert_colorbar(ax: Axes, im: ScalarMappable, **kwargs) -> ColorbarBase:
    """
    Insert colormap

    :param ax: ``Axes``
    :param im: ``ScalarMappable``
    :param kwargs: Additional args pass to ``ax.figure.colorbar``
    :return:
    """

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="7%", pad=0.1)

    return ax.figure.colorbar(im, cax=cax, **kwargs)


def insert_cyclic_colorbar(ax: Axes,
                           im: ScalarMappable,
                           *,
                           num_colors: int = 12,
                           num_labels: int = 4,
                           width: float = 0.6,
                           inner_diameter: float = 0.6,
                           vmin: float | None = None,
                           vmax: float | None = None) -> None:
    """
    Insert cyclic colormap in ``inset_axes``

    :param ax: ``Axes``
    :param im: ``ScalarMappable``
    :param num_colors: Number of color in the cyclic colorbar
    :param num_labels: Number of labels in the cyclic colorbar
    :param width: Width of the each color
    :param inner_diameter: The size of the inner circle
    :param vmin: Min value of the colormap, equal to ``vmax`` in cyclic data
    :param vmax: Max value of the colormap, equal to ``vmin`` in cyclic data
    """
    polar_ax = ax.inset_axes((1.1, 0.65, 0.4, 0.4), polar=True)
    theta = np.linspace(0, 2 * np.pi, num_colors, endpoint=False)
    r1 = np.ones_like(theta)
    r2 = np.ones_like(theta) * inner_diameter

    cmap = im.cmap.name
    cyclic_cmap = ['twilight', 'twilight_shifted', 'hsv']
    if cmap not in cyclic_cmap:
        raise ValueError(f'cmap should be one of the {cyclic_cmap}, not "{cmap}"')
    colors = get_customized_cmap(cmap, (0, 1), num_colors, endpoint=False)
    polar_ax.bar(theta, r1, color=colors, width=width, bottom=r2)

    # Add labels corresponding to the data values
    angles = np.linspace(0, 2 * np.pi, num_labels, endpoint=False)

    if vmin is None:
        vmin = im.norm.vmin
    if vmax is None:
        vmax = im.norm.vmax
    values = np.linspace(vmin, vmax, num_labels, endpoint=False)

    for angle, value in zip(angles, values):
        polar_ax.text(angle, 3, f'{value:.1f}', horizontalalignment='center', verticalalignment='center')

    polar_ax.set_yticklabels([])
    polar_ax.set_xticks([])  # remove angle labels
    polar_ax.grid(False)  # remove the grid
    polar_ax.set_frame_on(False)
