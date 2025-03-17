from pathlib import Path
from typing import Literal, Callable, Sequence

import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from neuralib.plot._dotplot import DotPlot
from neuralib.typing import ArrayLikeStr, ArrayLike
from neuralib.typing import PathLike, DataFrame
from neuralib.util.verbose import fprint
from scipy.stats import pearsonr

__all__ = [
    'dotplot',
    'scatter_histplot',
    'scatter_binx_plot',
    'axvline_histplot',
    'diag_histplot',
    'diag_heatmap',
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

def scatter_histplot(x: np.ndarray,
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
    sns.set_theme(style='white', font_scale=1.2)

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


# ============= #
# Diagonal Plot #
# ============= #


def diag_histplot(x: ArrayLike,
                  y: ArrayLike,
                  bins: int | Sequence[float] | str = 30,
                  *,
                  anchor: Literal['above', 'below'] = 'above',
                  hist_width: float = 0.3,
                  scatter_kws: dict | None = None,
                  polygon_kws: dict | None = None,
                  ax: Axes | None = None) -> None:
    """
    Scatter plot with an overlaid histogram along the diagonal

    `Dimension parameters`:

        N = number of sample points

    :param x: Numerical array. `Array[float, N]`
    :param y: Numerical array. `Array[float, N]`
    :param bins:  Number of bins (or bin specification) for ``np.histogram()``
    :param anchor: Whether the histogram bars extend above or below the main diagonal
    :param hist_width: Maximum length of histogram bars in the rotated space
    :param scatter_kws: Keyword arguments passed to ``ax.scatter()``
    :param polygon_kws: Keyword arguments passed to each ``Polygon()`` patch
    :param ax: ``Axes``
    :return:
    """
    if ax is None:
        _, ax = plt.subplots()

    x = np.array(x)
    y = np.array(y)

    if scatter_kws is None:
        scatter_kws = {'s': 3, 'c': 'black', 'marker': '.', 'edgecolors': 'none'}
    ax.scatter(x, y, **scatter_kws)

    #
    if polygon_kws is None:
        polygon_kws = {'facecolor': 'gray', 'edgecolor': 'none', 'zorder': -1}

    vmin = np.min([x, y])
    vmax = np.max([x, y])

    X = (x - y) / np.sqrt(2) * vmax  # Rotate by +45°
    Y_anchor = vmax / np.sqrt(2)

    # Compute histogram in rotated X coordinate (perpendicular to the anti-diagonal)
    counts, edges = np.histogram(X, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1

    sign = 1 if anchor == 'above' else -1

    R_inv = np.array([
        [1 / np.sqrt(2), 1 / np.sqrt(2)],
        [-1 / np.sqrt(2), 1 / np.sqrt(2)]
    ])

    for i in range(bins):
        c = counts[i]
        if c == 0:
            continue

        X0, X1 = edges[i], edges[i + 1]  # Bin edges in X (rotated coordinates)
        bar_height = (c / max_count) * hist_width  # Bar height is scaled by the bin count

        # Define the corners of the rectangle in (X, Y) space.
        # The rectangle is anchored at Y_anchor.
        # Its corners: (X0, Y_anchor), (X1, Y_anchor), (X1, Y_anchor + sign*bar_height), (X0, Y_anchor + sign*bar_height)
        corners_rot = np.array([
            [X0, X1, X1, X0],
            [Y_anchor, Y_anchor, Y_anchor + sign * bar_height, Y_anchor + sign * bar_height]
        ])

        # Transform the corners back to original (x,y) coordinates
        corners_xy = R_inv @ corners_rot  # shape (2,4)

        poly = Polygon(corners_xy.T, **polygon_kws)
        ax.add_patch(poly)

    ax.axline((0.5, 0.5), slope=1, color='k', alpha=0.5, lw=1)
    ax.set(xlim=(vmin, vmax), ylim=(vmin, vmax))
    ax.set_aspect('equal', adjustable='box')


def diag_heatmap(x: ArrayLike,
                 y: ArrayLike,
                 cmap: str = 'gist_earth',
                 *,
                 grid_xy: tuple[complex, complex] = (100j, 100j),
                 extent: tuple[float, float, float, float] | None = None,
                 scatter_kws: dict | None = None,
                 imshow_kws: dict | None = None,
                 ax: Axes | None = None) -> None:
    """
    Scatter plot with an overlaid kernel density heatmap

    :param x: Numerical array. `Array[float, N]`
    :param y: Numerical array. `Array[float, N]`
    :param cmap: Colormap used for the heatmap
    :param grid_xy: Grid specification for generating the heatmap
    :param extent: extent for the ``ax.imshow()``
    :param scatter_kws: Keyword arguments passed to ``ax.scatter()``
    :param imshow_kws: Keyword arguments passed to ``ax.imshow()``
    :param ax: ``Axes``
    :return:
    """
    if ax is None:
        _, ax = plt.subplots()

    x = np.array(x)
    y = np.array(y)

    if scatter_kws is None:
        scatter_kws = {'s': 3, 'c': 'gray', 'marker': '.', 'edgecolors': 'none'}
    ax.scatter(x, y, **scatter_kws)

    #
    if extent is None:
        x1, x2 = np.min(x), np.max(x)
        y1, y2 = np.min(y), np.max(y)
    else:
        x1, x2, y1, y2 = extent

    kde = _create_kde_map(x, y, grid_xy, extent=(x1, x2, y1, y2))

    if imshow_kws is None:
        imshow_kws = {}
    ax.imshow(np.rot90(kde), cmap=cmap, extent=(x1, x2, y1, y2), **imshow_kws)

    ax.axline((0.5, 0.5), slope=1, color='k', alpha=0.5, lw=1)
    ax.set_aspect('equal', adjustable='box')


def _create_kde_map(x, y, grid_xy, extent) -> np.ndarray:
    from scipy import stats

    if extent is None:
        x1, x2 = np.min(x), np.max(x)
        y1, y2 = np.min(y), np.max(y)
    else:
        x1, x2, y1, y2 = extent

    xstep, ystep = grid_xy

    X, Y = np.mgrid[x1:x2:xstep, y1:y2:ystep]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = stats.gaussian_kde(values)

    return np.reshape(kernel(positions).T, X.shape)


# ====== #
# Others #
# ====== #

def axvline_histplot(ax: Axes,
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


def grid_subplots(data: np.ndarray | list[np.ndarray],
                  images_per_row: int,
                  plot_func: Callable | str,
                  *,
                  dtype: Literal['xy', 'img'],
                  hide_axis: bool = True,
                  sharex: bool = False,
                  sharey: bool = False,
                  title: list[str] | None = None,
                  figsize: tuple[int, int] | None = None,
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
    :param title: List of title foreach show in the subplot
    :param figsize: Figure_size pass to ``plt.subplots()``
    :param output: Path to save the plot image. If None, displays the plot.
    :param kwargs: Additional keyword arguments passed to the plotting function ``plot_func``
    :return:
    """
    n_images = len(data)
    n_rows = np.ceil(n_images / images_per_row).astype(int)
    n_cols = min(images_per_row, n_images)

    if figsize is None:
        figsize = (n_cols, n_rows)

    _, ax = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False, sharex=sharex, sharey=sharey)

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

            # dtype
            if dtype == 'xy':
                dat = data[i]
                if dat.shape[1] != 2:
                    raise ValueError(f'invalid {data.size}')
                f(dat[:, 0], dat[:, 1], **kwargs)
            elif dtype == 'img':
                f(data[i], **kwargs)
            else:
                raise ValueError(f'unknown data type: {dtype}')

            # title
            if title is not None:
                ax[r, c].set_title(title[i])

        else:
            ax[r, c].set_visible(False)

    if output is None:
        plt.show()
    else:
        plt.savefig(output)
