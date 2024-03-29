from __future__ import annotations

from pathlib import Path
from typing import Literal, Callable

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.ndimage import gaussian_filter1d

from neuralib.plot.colormap import insert_colorbar
from neuralib.util.util_type import ArrayLike
from neuralib.util.util_verbose import fprint

__all__ = [
    'plot_2d_dots',
    'plot_regression_cc',
    'plot_joint_scatter_histogram',
    'plot_histogram_cutoff'
]


# ========= #
# Dots plot #
# ========= #

def plot_2d_dots(ax: Axes,
                 x: ArrayLike,
                 y: ArrayLike,
                 size: ArrayLike, *,
                 with_color: bool = False,
                 with_legends: bool = True,
                 size_type: Literal['area', 'radius'] = 'radius',
                 size_factor: float = 3000,
                 **kwargs):
    """

    :param ax:
    :param x:
    :param y:
    :param size:
    :param with_color:
    :param with_legends:
    :param size_type: whether the value corresponding to circle radius or surface area.
    :param size_factor:
    :param kwargs:
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
        ax_size_legend(ax, size, func)


def ax_size_legend(ax: Axes,
                   value: list[int] | np.ndarray,
                   f: Callable[[float], float] = None):
    """
    add the label and legend of the size in scatter plot

    :param ax:
    :param value: values reflect to size
    :param f: amplified callable
    :return:
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
                       bins=10,
                       **kwargs):
    """
    Regression to see the relationship between x and y

    :param ax:
    :param x:
    :param y:
    :param order:
    :param bins:
    :param show_cc:
    :param kwargs:
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
    _hist_line_plot(ax, x, y, bins)

    ax.set(**kwargs)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')

    if show_cc:
        cc = calculate_cc(x, y)
        ax.set_title(f'r = {round(cc, 4)}', fontstyle='italic')


def _hist_line_plot(ax,
                    x: np.ndarray,
                    y: np.ndarray,
                    bins=20,
                    ptype: Literal['median', 'mean'] = 'median'):
    """
    average or pickup the median of the y value in certain bins
    :param ax:
    :param x: (N, )
    :param y: (N, )
    :param bins:
    :param ptype
    :return:
    """

    if ptype == 'mean':

        a, edg = np.histogram(x, weights=y, bins=bins)
        n = np.histogram(x, bins)[0]  # (B, )

        weights = np.divide(a, n, out=np.zeros_like(a, dtype=float), where=n != 0)  # avoid true divide
        weights = gaussian_filter1d(weights, 1)

        ax.plot(edg[1:], weights, 'r--', alpha=0.5)

    elif ptype == 'median':
        counts, edg = np.histogram(x, weights=y, bins=bins)

        medians = []
        for i in range(len(counts)):
            bin_data = y[(x >= edg[i]) & (x < edg[i + 1])]
            median_val = np.nanmedian(bin_data)
            medians.append(median_val)

        ax.plot(edg[1:], medians, 'r--', alpha=0.5)

    else:
        raise ValueError(f'{ptype}')


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

    :param ax:
    :param value: (N,) 1d array
    :param cutoff: cutoff (threshold) value for the certain value, >= represents pass
    :param mask: (N,) mask for value. i.e., cell selection
    :param kwargs: passed to `ax.set`
    """
    if value.ndim != 1:
        raise ValueError('value must be 1d array')

    if mask is not None:
        value = value[mask]

    if cutoff > np.max(value) or cutoff < np.min(value):
        fprint(f'{cutoff} should be within {np.min(value)} and {np.max(value)}', vtype='warning')

    # ax.hist(value, 50, density=True, color='grey')
    sns.histplot(value, bins=30, kde=True, ax=ax, color='grey', stat='percent', element='step')
    ax.axvline(cutoff, color='r', linestyle='--', zorder=1)
    ax.set(**kwargs)
    ax.set_aspect(1.0 / ax.get_data_ratio(), adjustable='box')


def plot_joint_scatter_histogram(x: np.ndarray,
                                 y: np.ndarray,
                                 show_cc: bool = True,
                                 output: Path | None = None,
                                 **kwargs):
    """plot the linear correlation scatter and histogram between two variables"""

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
