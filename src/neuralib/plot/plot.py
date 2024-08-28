from __future__ import annotations

from pathlib import Path
from typing import Literal, Callable, Sequence

import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.stats import pearsonr

from neuralib.plot._dotplot import DotPlot
from neuralib.plot.colormap import insert_colorbar
from neuralib.typing import ArrayLike, ArrayLikeStr
from neuralib.typing import PathLike, DataFrame
from neuralib.util.deprecation import deprecated_aliases, deprecated_func
from neuralib.util.verbose import fprint

__all__ = [
    'dotplot',
    'plot_2d_dots',
    'scatter_binx_plot',
    'scatter_histogram',
    'hist_cutoff',
    'violin_boxplot',
    'grid_subplots',
]


# ========= #
# Dots plot #
# ========= #

def dotplot(xlabel: ArrayLikeStr,
            ylabel: ArrayLikeStr,
            values: np.ndarray,
            *,
            scale: Literal['area', 'radius'] = 'area',
            max_marker_size: float | None = None,
            size_title: str | None = None,
            size_legend_num: int | None = None,
            size_legend_as_int: bool = True,
            with_color: bool = False,
            cmap: mcolors.Colormap = 'Reds',
            colorbar_title: str | None = None,
            norm: mcolors.Normalize | None = None,
            cbar_vmin: float | None = None,
            cbar_vmax: float | None = None,
            figure_title: str | None = None,
            figure_output: PathLike | None = None,
            ax: Axes | None = None,
            **kwargs):
    """
     Plot values as dots, with the option also in colormap

    `Dimension parameters`:

        X = number of x label

        Y = number of y label

    :param xlabel: String arraylike. `ArrayLike[str, X]`
    :param ylabel: String arraylike. `ArrayLike[str, X]`
    :param values: 2D value array. `ArrayLike[str, [X, Y]]`
    :param scale: Dot size representation. {'area', 'radius'}
    :param max_marker_size: Marker size for the max value
    :param size_title: Size_title in the size legend
    :param size_legend_num: Number of legend to be shown
    :param size_legend_as_int: Size legend show only `Int`
    :param with_color: If dot with colormap
    :param cmap: ``Colormap``
    :param colorbar_title: Title of the colorbar
    :param norm: Colorbar Normalize
    :param cbar_vmin: Value min for the colorbar
    :param cbar_vmax: Value nax for the colorbar
    :param figure_title: Figure title
    :param figure_output: Figure save output path
    :param ax: If existing axes ``Axes``
    :param kwargs: additional arguments to ``ax.scatter()``
    """

    dp = DotPlot(xlabel, ylabel, values,
                 scale=scale,
                 max_marker_size=max_marker_size,
                 size_title=size_title,
                 size_legend_num=size_legend_num,
                 size_legend_as_int=size_legend_as_int,
                 with_color=with_color,
                 cmap=cmap,
                 colorbar_title=colorbar_title,
                 norm=norm,
                 cbar_vmin=cbar_vmin,
                 cbar_vmax=cbar_vmax,
                 figure_title=figure_title,
                 figure_output=figure_output,
                 ax=ax)

    dp.plot(**kwargs)


@deprecated_aliases(x='xlabel', y='ylabel', size='values')
@deprecated_func(new='neuralib.plot.dotplot()', removal_version='0.3')
def plot_2d_dots(ax: Axes,
                 xlabel: ArrayLike,
                 ylabel: ArrayLike,
                 values: np.ndarray, *,
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
    :param xlabel: string values. `ArrayLike[str, X]`
    :param ylabel: string values. `ArrayLike[str, X]`
    :param values: size of the dots. `Array(float, [X, Y]]`
    :param with_color: `size` domain as colormap
    :param with_legends: show the value scaling as legend
    :param size_type: whether the value corresponding to circle radius or surface area.
    :param size_factor: scaling factor for visualization
    :param kwargs: passed to ``ax.scatter()``
    :return:
    """
    values = np.array([values]).flatten()
    xlabel, ylabel = np.meshgrid(xlabel, ylabel, indexing='ij')

    #

    def _func(v):
        if size_type == 'radius':
            return v ** 2 * size_factor
        elif size_type == 'area':
            return v * size_factor
        else:
            raise ValueError('')

    s = _func(values)
    #
    if with_color:
        im = ax.scatter(xlabel.ravel(), ylabel.ravel(), s=s, c=values, cmap='viridis', clip_on=False,
                        vmin=0, vmax=1, **kwargs)
        insert_colorbar(ax, im)
    else:
        ax.scatter(xlabel.ravel(), ylabel.ravel(), s=s, c='k', clip_on=False, **kwargs)

    if with_legends:
        _ax_size_legend(ax, values, _func)


def _ax_size_legend(ax: Axes,
                    value: list[int] | np.ndarray,
                    f: Callable[[float], float]):
    """
    add the labels and legend of the size in scatter plot.

    :param ax: ``Axes``
    :param value: values reflect to size
    :param f: amplified callable
    """
    vsize = np.linspace(0, np.max(value), num=5)

    for s in vsize:
        ax.scatter([], [], s=f(s), c='k', label=str(s))

    handles, labels = plt.gca().get_legend_handles_labels()

    plt.legend(handles[1:], labels[1:], labelspacing=1.2, title="value", borderpad=1,
               frameon=True, framealpha=0.6, edgecolor="k", facecolor="w", loc='right')


# ========== #
# Regression #
# ========== #

@deprecated_func(new='scatter_binx_plot()', removal_version='0.3')
def plot_regression_cc(*args, **kwargs):
    scatter_binx_plot(*args, **kwargs)


@deprecated_aliases(show_cc='linear_reg')
def scatter_binx_plot(ax: Axes,
                      x: np.ndarray,
                      y: np.ndarray,
                      bins: int | Sequence[float] | str = 10,
                      *,
                      order: int = 1,
                      linear_reg: bool = True,
                      bin_func: Literal['median', 'mean'] = 'median',
                      **kwargs):
    """
    Regression to see the relationship between x and y

    `Dimension parameters`:

        N = number of sample points

    :param ax: ``Axes``
    :param x: Numerical array. `Array[float, N]`
    :param y: Numerical array. `Array[float, N]`
    :param bins: passed to ``numpy.histogram()``
    :param order: If order is greater than `1`, use numpy.polyfit to estimate a polynomial regression
    :param linear_reg: Show linear correlation coefficient
    :param bin_func: Literal['median', 'mean']. default is median
    :param kwargs: additional args passed to ``ax.set``
    :return:
    """
    import seaborn as sns

    sns.regplot(
        x=x,
        y=y,
        order=order,
        scatter_kws={'alpha': 0.5,
                     'color': 'grey',
                     's': 8,
                     'edgecolors': 'none'},
        line_kws={'color': 'black'},
        ax=ax
    )
    _hist_line_plot(ax, x, y, bins, bin_func=bin_func)

    ax.set(**kwargs)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    if linear_reg:
        cc = pearsonr(x, y)[0]
        ax.set_title(f'r = {round(cc, 4)}', fontstyle='italic')


def _hist_line_plot(ax,
                    x: np.ndarray,
                    y: np.ndarray,
                    bins: int | Sequence[float] | str = 20,
                    bin_func: Literal['median', 'mean'] = 'median'):
    """
    average or pickup the median of the y value in certain bins
    :param ax: ``Axes``
    :param x: `Array[float, N]`
    :param y: `Array[float, N]`
    :param bins:  passed to ``numpy.histogram()``
    :param bin_func
    :return:
    """

    hist, edg = np.histogram(x, weights=y, bins=bins)

    if bin_func == 'mean':
        n = np.histogram(x, bins)[0]  # (B, )
        bin_values = np.divide(hist, n, out=np.zeros_like(hist, dtype=float), where=n != 0)  # avoid true divide
    elif bin_func == 'median':
        bin_values = []
        for i in range(len(hist)):
            bin_data = y[(x >= edg[i]) & (x < edg[i + 1])]
            median_val = np.nanmedian(bin_data)
            bin_values.append(median_val)
    else:
        raise ValueError(f'{bin_func}')

    x = edg[:-1] + np.median(np.diff(edg)) / 2  # alignment
    ax.plot(x, bin_values, 'r--', alpha=0.5)


# ====== #
# Others #
# ====== #


def hist_cutoff(ax: Axes,
                values: np.ndarray,
                cutoff: float,
                bins: int = 30,
                *,
                mask: np.ndarray | None = None,
                **kwargs) -> None:
    """
    Plot the histogram with a cutoff value

    :param ax: ``Axes``
    :param values: 1d array. `Array[float, N]`
    :param cutoff: cutoff (threshold) value for the certain value, >= represents pass
    :param bins: passed to ``numpy.histogram()``
    :param mask: mask for value. i.e., cell selection. `Array[bool, N]`
    :param kwargs: passed to ``ax.set``
    """
    if values.ndim != 1:
        raise ValueError('value must be 1d array')

    if mask is not None:
        values = values[mask]

    if cutoff > np.max(values) or cutoff < np.min(values):
        fprint(f'{cutoff} should be within {np.min(values)} and {np.max(values)}', vtype='warning')

    sns.histplot(values, bins=bins, kde=True, color='grey', stat='percent', element='step', ax=ax)
    ax.axvline(cutoff, color='r', linestyle='--', zorder=1)
    ax.set(**kwargs)


@deprecated_func(new='scatter_histogram()', removal_version='0.3')
def plot_joint_scatter_histogram(*args, **kwargs):
    scatter_histogram(*args, **kwargs)


@deprecated_aliases(show_cc='linear_reg')
def scatter_histogram(x: np.ndarray,
                      y: np.ndarray,
                      bins: int | Sequence[float] | str = 15,
                      *,
                      linear_reg: bool = True,
                      output: Path | None = None,
                      **kwargs):
    """
    plot the linear correlation scatter and histogram between two variables

    `Dimension parameters`:

        N = number of sample points

    :param x: numerical array x. `Array[float, N]`
    :param y: numerical array y. `Array[float, N]`
    :param bins: passed to ``numpy.histogram()``
    :param linear_reg: If show correlation coefficient
    :param output: Figure save output
    :param kwargs: additional args pass through ``ax.set()``
    :return:
    """

    sns.set(style='white', font_scale=1.2)

    g = sns.JointGrid(x=x, y=y, height=5)

    if linear_reg:
        g.plot_joint(sns.regplot, color='grey')
    else:
        g.plot_joint(sns.scatterplot, color='grey')

    g = g.plot_marginals(sns.histplot, kde=False, bins=bins, color='grey')

    ax = g.ax_joint
    ax.set(**kwargs)

    #
    if linear_reg:
        cc = pearsonr(x, y)[0]
        ax.text(0.5, 0.8, f'r = {round(cc, 4)}', fontstyle='italic',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)

    if output is not None:
        plt.savefig(output)
    else:
        plt.show()


@deprecated_func(new='violin_boxplot()', removal_version='0.3')
def plot_half_violin_box_dot(*args, **kwargs):
    violin_boxplot(*args, **kwargs)


def violin_boxplot(ax: Axes,
                   data: DataFrame | dict | list[np.ndarray],
                   x: str | None = None,
                   y: str | None = None,
                   hue: str | None = None,
                   scatter_alpha: float = 0.7,
                   scatter_size: float = 3,
                   output: PathLike | None = None,
                   **kwargs) -> None:
    """
    Plot the data with half violin together with boxes and scatters

    :param ax: ``Axes``
    :param data: Dataset for plotting
    :param x: Names of variables in data or vector data: ``x``
    :param y: Names of variables in data or vector data: ``y``
    :param hue: Names of variables in data or vector data: ``hue``
    :param scatter_alpha: Scatter alpha for the ``sns.stripplot()``
    :param scatter_size: Scatter size for the ``sns.stripplot()``
    :param output: Fig save output path
    :param kwargs: Common args pass through ``sns.violinplot()``, ``sns.boxplot()`` and ``sns.stripplot()``
    :return:
    """
    kwargs = dict(
        ax=ax,
        x=x,
        y=y,
        hue=hue,
        data=data,
        **kwargs
    )

    sns.violinplot(dodge=False, density_norm="width", inner=None, **kwargs)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData))

    sns.boxplot(saturation=1,
                showfliers=False,
                width=0.3,
                boxprops={'zorder': 3, 'facecolor': 'none'},
                **kwargs)

    old_len_collections = len(ax.collections)

    sns.stripplot(dodge=False,
                  alpha=scatter_alpha,
                  size=scatter_size,
                  **kwargs)

    for dots in ax.collections[old_len_collections:]:
        dots.set_offsets(dots.get_offsets() + np.array([0.12, 0]))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    if output is not None:
        plt.savefig(output)


@deprecated_func(new='grid_subplots()', removal_version='0.3')
def plot_grid_subplots(*args, **kwargs):
    grid_subplots(*args, **kwargs)


def grid_subplots(data: np.ndarray | list[np.ndarray],
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
        >>> grid_subplots(data, 5, 'plot', dtype='xy')

    Example for plot img array grid ::

        >>> data = np.random.sample((30, 10, 10))
        >>> grid_subplots(data, 5, 'imshow', dtype='img', cmap='gray')

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
