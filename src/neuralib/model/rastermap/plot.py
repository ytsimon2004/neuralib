from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
from typing import NamedTuple, Literal
from typing_extensions import Self

from neuralib.model.rastermap import RasterMapResult
from neuralib.plot import plot_figure
from neuralib.typing import PathLike
from neuralib.util.deprecation import deprecated_class

__all__ = [
    'plot_rastermap',
    'plot_cellular_spatial',
    'plot_wfield_spatial',
    'RasterMapPlot',
    'Covariant'
]


def plot_rastermap(result: RasterMapResult,
                   act_time: np.ndarray, *,
                   time_range: tuple[float, float] | None = None,
                   covars: list[Covariant] | None = None,
                   output: PathLike | None = None):
    """
    plot the rastermap result with behavioral measurements

    :param result: :class:`~.core.RasterMapResult`
    :param act_time: neural activity time array. should be the same T as neural_activity when run the rastermap
    :param time_range: time range for plotting (START,END)
    :param covars: list of :class:`~Covariant`
    :param output: output path for figure save. If None then show
    """
    plotter = RasterMapPlot(result, act_time, time_range, covars, output)
    plotter.plot_rastermap()


def plot_cellular_spatial(result: RasterMapResult,
                          xpos: np.ndarray,
                          ypos: np.ndarray,
                          ax: Axes | None = None,
                          output: PathLike | None = None,
                          **kwargs):
    """
    Plot spatial location of each cell cluster by rastermap

    :param result: :class:`~.core.RasterMapResult`
    :param xpos: soma central X position.`Array[float, N]`
    :param ypos: soma central Y position.`Array[float, N]`
    :param ax: ``Axes``
    :param output: output path for figure save. If None then show
    :param kwargs: additional arguments pass to ``ax.set()``
    :return:
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(xpos, ypos, s=8, c=result.embedding, cmap="gist_ncar", alpha=0.25)
    ax.invert_yaxis()
    ax.set(**kwargs)
    ax.set_aspect('equal')

    if output is not None:
        plt.savefig(output)
    else:
        plt.show()


def plot_wfield_spatial(result: RasterMapResult,
                        width: int,
                        height: int,
                        ax: Axes | None = None,
                        output: PathLike | None = None,
                        **kwargs):
    """
    Plot spatial location of each pixel cluster by rastermap

    :param result: :class:`~.core.RasterMapResult`
    :param width: sequence image width
    :param height: sequence image height
    :param ax: ``Axes``
    :param output: output path for figure save. If None then show
    :param kwargs: additional arguments pass to ``ax.set()``
    """

    if ax is None:
        _, ax = plt.subplots()

    x = np.arange(width)
    y = np.arange(height)
    xpos, ypos = np.meshgrid(x, y)  # Array[float, [W, H]]

    ax.scatter(xpos, ypos, s=1, c=result.embedding, cmap="gist_ncar", alpha=0.25)
    ax.invert_yaxis()
    ax.set(**kwargs)
    ax.set_aspect('equal')

    if output is not None:
        plt.savefig(output)
    else:
        plt.show()


class RasterMapPlot:
    """Plot the rastermap result with behavioral measurements"""

    DEFAULT_TRIAL_COLORS = ('royalblue', 'darkorange', 'seagreen', 'crimson')

    def __init__(self, result: RasterMapResult,
                 act_time: np.ndarray,
                 time_range: tuple[float, float] | None = None,
                 covars: list[Covariant] | None = None,
                 event_colors: list[str] | None = None,
                 output: PathLike | None = None):
        """
        :param result: class:`~.core.RasterMapResult`
        :param act_time: neural activity time array. should be the same T as neural_activity when run the rastermap
        :param time_range: time range for plotting (START,END)
        :param covars: list of :class:`~Covariant`
        """
        self.raster = result
        self.covars = covars

        self.time_range = time_range or (act_time[0], act_time[-1])

        if time_range is not None:
            self.act_mask = np.logical_and(time_range[0] < act_time, act_time < time_range[1])
        else:
            self.act_mask = np.ones_like(act_time, dtype=np.bool_)

        self.act_time = act_time[self.act_mask]

        self.event_colors = event_colors or self.DEFAULT_TRIAL_COLORS
        self.output = output

    @property
    def super_neurons(self) -> np.ndarray:
        """rastermap sorted 2D array. `Array[float, [N, T]]`"""
        return self.raster.super_neurons[:, self.act_mask]

    def process_continuous(self) -> list[Covariant]:
        """process behavioral measurements, select time range and do the interpolation same shape as neural activity"""
        return [
            cov.masking_time(self.time_range).interp_activity(self.act_time)
            for cov in self.covars
            if cov.dtype == 'continuous'
        ]

    def plot_rastermap(self):
        if self.covars is not None:
            covars = self.process_continuous()
            n_covars = len(covars)
        else:
            covars = None
            n_covars = 1

        height_ratios = [1] * n_covars + [7]
        with plot_figure(self.output, n_covars + 1, 1, gridspec_kw={'height_ratios': height_ratios},
                         tight_layout=False, sharex=True) as _ax:

            # continuous dtype
            if self.covars is not None:
                for i, cov in enumerate(covars):
                    if cov.dtype == 'continuous':
                        ax = _ax[i]
                        ax.plot(cov.time, cov.value, color='k')
                        ax.set_xlim(self.time_range)
                        ax.axis('off')
                        ax.set_title(cov.name)
            else:
                _ax[0].axis('off')

            # rastermap
            ax = _ax[n_covars]
            ax.imshow(
                self.super_neurons,
                cmap='gray_r',
                vmin=0,
                vmax=0.8,
                aspect='auto',
                interpolation='none',
                extent=(self.time_range[0], self.time_range[1], self.raster.n_clusters, 0),
            )
            ax.set(xlabel='time', ylabel='rastermap clusters')

            # event segments
            if self.covars is not None:
                self.plot_segments(ax=ax)

            # colormap
            n_clusters = self.raster.n_clusters
            cluster_colors = plt.get_cmap("gist_ncar", n_clusters)
            cb_ax = inset_axes(ax, width="2%", height="100%", loc="right",
                               bbox_to_anchor=(0.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)

            cb_ax.imshow(
                np.arange(n_clusters)[:, np.newaxis],
                cmap=cluster_colors,
                aspect="auto"
            )
            cb_ax.axis("off")

    def plot_segments(self, ax: Axes):
        """
        Plot event segments as vertical spans on the axis

        :param ax: matplotlib Axes object
        """
        legend_patches = []

        for i, cov in enumerate(self.covars):
            if cov.dtype == 'event':
                for start, end in cov.time:
                    if self.time_range is not None:
                        t0 = self.time_range[0]
                        t1 = self.time_range[1]
                        if end < t0 or start > t1:
                            continue  # completely outside view
                        start = max(start, t0)
                        end = min(end, t1)

                    ax.axvspan(start, end, color=self.event_colors[i], alpha=0.4)

                legend_patches.append(Patch(facecolor=self.event_colors[i], alpha=0.4, label=cov.name))

        if legend_patches:
            ax.legend(handles=legend_patches, loc='upper right', fontsize=10, frameon=False)


class Covariant(NamedTuple):
    """
    Covariant variable that can be plotted alongside rastermap results.

    Supports two types:

    - ``'continuous'``: time-series data (e.g., velocity, position)
    - ``'event'``: discrete time segments (e.g., trial periods, behavioral events)

    :param name: name of the covariant variable
    :param dtype: type of the covariant variable (``'event'`` or ``'continuous'``)
    :param time: time array. ``Array[float, T]`` for continuous dtype or ``Array[float, [E, 2]]`` for on/off event dtype
    :param value: value array (only for continuous dtype). ``Array[float, T]``
    """
    name: str
    dtype: Literal['event', 'continuous']
    time: np.ndarray
    value: np.ndarray | None = None

    def masking_time(self, t: tuple[float, float]) -> Self:
        """
        Mask data to a specific time range (continuous dtype only).

        :param t: (START,END) time range
        :return: new Covariant with data filtered to the time range
        :raises ValueError: if called on event dtype
        """
        if self.dtype == 'event':
            raise ValueError('method only available for continuous dtype')

        mx = np.logical_and(t[0] < self.time, self.time < t[1])
        return self._replace(time=self.time[mx], value=self.value[mx])

    def interp_activity(self, act_time: np.ndarray) -> Self:
        """
        Interpolate data to match another activity time array (continuous dtype only).

        :param act_time: activity time array to interpolate to. `Array[float, T']`
        :return: new Covariant with interpolated values matching act_time
        :raises ValueError: if called on event dtype
        """
        if self.dtype == 'event':
            raise ValueError('method only available for continuous dtype')

        v = interp1d(self.time, self.value, bounds_error=False, fill_value=0)(act_time)
        return self._replace(time=act_time, value=v)


@deprecated_class(new='neuralib.model.rastermap.plot.Covariant', removal_version='0.6.0')
class BehavioralVT(NamedTuple):
    name: str
    """name of the behavioral variable"""
    time: np.ndarray
    """time array. `Array[float, T]`"""
    value: np.ndarray
    """value array. `Array[float, T]`"""

    def masking_time(self, t: tuple[float, float]) -> Self:
        """
        mask given time range
        :param t: (START,END) time range
        :return:
        """
        mx = np.logical_and(t[0] < self.time, self.time < t[1])
        return self._replace(time=self.time[mx], value=self.value[mx])

    def interp_activity(self, act_time: np.ndarray) -> Self:
        """
        interpolation to another activity array. i.e., neural activity
        :param act_time: activity array. `Array[float, T']`
        :return:
        """
        v = interp1d(self.time, self.value, bounds_error=False, fill_value=0)(act_time)
        return self._replace(time=act_time, value=v)
