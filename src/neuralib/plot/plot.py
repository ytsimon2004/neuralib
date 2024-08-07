from __future__ import annotations

from pathlib import Path
from typing import Literal, Callable

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter1d

from neuralib.plot.colormap import insert_colorbar
from neuralib.typing import ArrayLike
from neuralib.typing import PathLike, DataFrame
from neuralib.util.verbose import fprint

__all__ = [
    'plot_2d_dots',
    'plot_regression_cc',
    'plot_joint_scatter_histogram',
    'plot_histogram_cutoff',
    'plot_half_violin_box_dot',
    'plot_grid_subplots'
]


# ========= #
# Dots plot #
# ========= #

def plot_2d_dots(ax: Axes,
                 x: ArrayLike,
                 y: ArrayLike,
                 size: np.ndarray, *,
                 with_color: bool = False,
                 with_legends: bool = True,
                 size_type: Literal['area', 'radius'] = 'radius',
                 size_factor: float = 3000,
                 **kwargs):
    """
    Plot values as 2D dots

    `Dimension parameters`:

        X = number of x label

        Y = number of y label

    :param ax: ``Axes``
    :param x: string values. `ArrayLike[str, X]`
    :param y: string values. `ArrayLike[str, X]`
    :param size: size of the dots. `Array(float, [Y, X]]`
    :param with_color: `size` domain as colormap
    :param with_legends: show the value scaling as legend
    :param size_type: whether the value corresponding to circle radius or surface area.
    :param size_factor: scaling factor for visualization
    :param kwargs: passed to ``ax.scatter()``
    :return:
    """
    size = np.array([size]).flatten()
    x, y = np.meshgrid(x, y, indexing='ij')
    #
    if size_type == 'radius':
        func = lambda it: it ** 2 * size_factor
    elif size_type == 'area':
        func = lambda it: it * size_factor
    else:
        raise ValueError('')

    s = func(size)
    #
    if with_color:
        im = ax.scatter(x.ravel(), y.ravel(), s=s, c=size, cmap='viridis', clip_on=False,
                        vmin=0, vmax=1, **kwargs)
        insert_colorbar(ax, im)
    else:
        ax.scatter(x.ravel(), y.ravel(), s=s, c='k', clip_on=False, **kwargs)

    if with_legends:
        _ax_size_legend(ax, size, func)


def _ax_size_legend(ax: Axes,
                    value: list[int] | np.ndarray,
                    f: Callable[[float], float] = None):
    """
    add the label and legend of the size in scatter plot. TODO as int legend

    :param ax: ``Axes``
    :param value: values reflect to size
    :param f: amplified callable
    """
    if f is None:
        f = lambda it: (it ** 2) * 3000

    vsize = np.linspace(0, np.max(value), num=5)

    for s in vsize:
        ax.scatter([], [], s=f(s), c='k', label=str(s))

    h, l = plt.gca().get_legend_handles_labels()

    plt.legend(h[1:], l[1:], labelspacing=1.2, title="value", borderpad=1,
               frameon=True, framealpha=0.6, edgecolor="k", facecolor="w")


# ========== #
# Regression #
# ========== #


def plot_regression_cc(ax: Axes,
                       x: np.ndarray,
                       y: np.ndarray,
                       order: int = 1,
                       show_cc: bool = True,
                       bins: int = 10,
                       bin_func: Literal['median', 'mean'] = 'median',
                       **kwargs):
    """
    Regression to see the relationship between x and y

    `Dimension parameters`:

        N = number of sample points

    :param ax: ``Axes``
    :param x: numerical array. `Array[float, N]`
    :param y: numerical array. `Array[float, N]`
    :param order: order of the polynomial to fit when calculating the residuals.
    :param bins: number of bins
    :param show_cc: show correlation coefficient
    :param bin_func: Literal['median', 'mean']. default is median
    :param kwargs: additional args passed to ``ax.set``
    :return:
    """
    import seaborn as sns

    sns.regplot(x=x, y=y, ax=ax, order=order,
                scatter_kws={
                    'alpha': 0.5,
                    'color': 'grey',
                    's': 8,
                    'edgecolors': 'none'
                },
                line_kws={
                    'color': 'black'
                })
    _hist_line_plot(ax, x, y, bins, bin_func=bin_func)

    ax.set(**kwargs)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    if show_cc:
        cc = calculate_cc(x, y)
        ax.set_title(f'r = {round(cc, 4)}', fontstyle='italic')


def _hist_line_plot(ax,
                    x: np.ndarray,
                    y: np.ndarray,
                    bins: int = 20,
                    bin_func: Literal['median', 'mean'] = 'median'):
    """
    average or pickup the median of the y value in certain bins
    :param ax: ``Axes``
    :param x: `Array[float, N]`
    :param y: `Array[float, N]`
    :param bins:
    :param bin_func
    :return:
    """

    if bin_func == 'mean':
        a, edg = np.histogram(x, weights=y, bins=bins)
        n = np.histogram(x, bins)[0]  # (B, )

        weights = np.divide(a, n, out=np.zeros_like(a, dtype=float), where=n != 0)  # avoid true divide
        weights = gaussian_filter1d(weights, 1)

        x = edg[:-1] + np.median(np.diff(edg)) / 2  # alignment
        ax.plot(x, weights, 'r--', alpha=0.5)

    elif bin_func == 'median':
        counts, edg = np.histogram(x, weights=y, bins=bins)

        medians = []
        for i in range(len(counts)):
            bin_data = y[(x >= edg[i]) & (x < edg[i + 1])]
            median_val = np.nanmedian(bin_data)
            medians.append(median_val)

        x = edg[:-1] + np.median(np.diff(edg)) / 2  # alignment
        ax.plot(x, medians, 'r--', alpha=0.5)

    else:
        raise ValueError(f'{bin_func}')


def calculate_cc(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import pearsonr
    try:
        cc = pearsonr(x, y)[0]
    except ValueError as e:
        print('input shape ERROR:', x.shape, y.shape)
        raise e

    return cc


# ====== #
# Others #
# ====== #


def plot_histogram_cutoff(ax: Axes,
                          value: ArrayLike,
                          cutoff: float,
                          mask: np.ndarray | None = None,
                          **kwargs) -> None:
    """
    Plot the histogram with a cutoff value

    :param ax: ``Axes``
    :param value: 1d array. `Array[float, N]`
    :param cutoff: cutoff (threshold) value for the certain value, >= represents pass
    :param mask: mask for value. i.e., cell selection. `Array[bool, N]`
    :param kwargs: passed to ``ax.set``
    """
    if value.ndim != 1:
        raise ValueError('value must be 1d array')

    if mask is not None:
        value = value[mask]

    if cutoff > np.max(value) or cutoff < np.min(value):
        fprint(f'{cutoff} should be within {np.min(value)} and {np.max(value)}', vtype='warning')

    sns.histplot(value, bins=30, kde=True, ax=ax, color='grey', stat='percent', element='step')
    ax.axvline(cutoff, color='r', linestyle='--', zorder=1)
    ax.set(**kwargs)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')


def plot_joint_scatter_histogram(x: np.ndarray,
                                 y: np.ndarray,
                                 show_cc: bool = True,
                                 output: Path | None = None,
                                 **kwargs):
    """
    plot the linear correlation scatter and histogram between two variables

    `Dimension parameters`:

        N = number of sample points

    :param x: numerical array x. `Array[float, N]`
    :param y: numerical array y. `Array[float, N]`
    :param show_cc: if show correlation coefficient
    :param output: fig save output
    :param kwargs: additional args pass through ``ax.set()``
    :return:
    """

    sns.set(style='white', font_scale=1.2)

    g = sns.JointGrid(x=x, y=y, height=5)
    g = g.plot_joint(sns.regplot, color='grey')
    g = g.plot_marginals(sns.histplot, kde=False, bins=12, color='grey')

    ax = g.ax_joint
    ax.set(**kwargs)

    if show_cc:
        cc = calculate_cc(x, y)
        ax.text(0.5, 0.8, f'r = {round(cc, 4)}', fontstyle='italic',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)

    if output is not None:
        plt.savefig(output)
    else:
        plt.show()


def plot_half_violin_box_dot(ax: Axes,
                             data: DataFrame | dict | list[np.ndarray],
                             x: str | None = None,
                             y: str | None = None,
                             hue: str | None = None,
                             output: PathLike | None = None,
                             **kwargs) -> None:
    """
    Plot the data with half violin together with boxes and dots

    :param ax: ``Axes``
    :param data: Dataset for plotting
    :param x: names of variables in data or vector data: ``x``
    :param y: names of variables in data or vector data: ``y``
    :param hue: names of variables in data or vector data: ``hue``
    :param output: fig save output
    :param kwargs: pass through ``sns.violinplot``, ``sns.boxplot`` and ``sns.stripplot``
    :return:
    """
    kws = dict(
        ax=ax,
        x=x,
        y=y,
        hue=hue,
        data=data,
        palette='Set2',
        **kwargs
    )

    sns.violinplot(dodge=False, density_norm="width", inner=None, **kws)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))

    sns.boxplot(saturation=1, showfliers=False, width=0.3, boxprops={'zorder': 3, 'facecolor': 'none'}, **kws)

    old_len_collections = len(ax.collections)
    sns.stripplot(dodge=False, alpha=0.7, size=3, **kws)

    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if output is not None:
        plt.savefig(output)


def plot_grid_subplots(data: np.ndarray | list[np.ndarray],
                       images_per_row: int,
                       plot_func: Callable | str, *,
                       dtype: Literal['xy', 'img'],
                       hide_axis: bool = True,
                       sharex: bool = False,
                       sharey: bool = False,
                       output: PathLike | None = None,
                       **kwargs) -> None:
    r"""
    Plots a sequence of subplots in a grid format

    Example for plot xy grid::

        >>> data = np.random.sample((30, 10, 2))
        >>> plot_grid_subplots(data, 5, 'plot', dtype='xy')

    Example for plot img array grid ::

        >>> data = np.random.sample((30, 10, 10))
        >>> plot_grid_subplots(data, 5, 'imshow', dtype='img', cmap='gray')

    :param data: 3D Array containing the data to be plotted. For 'xy' dtype, the shape must be (N, (\*, 2)).
        For 'img' dtype, the shape must be (N, (\*img)). Accepted also list of 2D array different size
    :param images_per_row: Number of images per row in the subplot grid
    :param plot_func: Function or method name to be used for plotting. If a string is provided,
        it should be a valid method name of a matplotlib Axes object
    :param dtype: {'xy', 'img'}. Type of data. 'xy' for (x, y) coordinate data, 'img' for image data
    :param hide_axis: If True, hides the axes of the subplots
    :param sharex: sharex acrross grid plots
    :param sharey: sharey acrross grid plots
    :param output: Path to save the plot image. If None, displays the plot.
    :param kwargs: Additional keyword arguments passed to the plotting function ``plot_func``
    :return:
    """
    n_images = len(data)
    n_rows = np.ceil(n_images / images_per_row).astype(int)
    n_cols = min(images_per_row, n_images)

    _, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), squeeze=False, sharex=sharex, sharey=sharey)

    for i in range(n_rows * n_cols):
        r, c = divmod(i, images_per_row)
        if hide_axis:
            ax[r, c].axis('off')

        #
        if i < n_images:  # check to avoid index error
            if callable(plot_func):
                f = plot_func
                kwargs['ax'] = ax[r, c]
            else:
                f = getattr(ax[r, c], plot_func)

            #
            if dtype == 'xy':
                dat = data[i]
                if dat.shape[1] != 2:
                    raise ValueError(f'invalid {data.size}')
                f(dat[:, 0], dat[:, 1], **kwargs)
            elif dtype == 'img':
                f(data[i], **kwargs)
            else:
                raise ValueError(f'unknown data type: {dtype}')
        else:
            ax[r, c].set_visible(False)

    if output is None:
        plt.show()
    else:
        plt.savefig(output)
