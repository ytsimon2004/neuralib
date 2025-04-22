from __future__ import annotations

from typing import NamedTuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
from typing_extensions import Self

from neuralib.model.rastermap import RasterMapResult
from neuralib.plot import plot_figure
from neuralib.typing import PathLike

__all__ = [
    'plot_rastermap',
    'plot_cellular_spatial',
    'plot_wfield_spatial',
    'RasterMapPlot',
    'BehavioralVT'
]


def plot_rastermap(result: RasterMapResult,
                   act_time: np.ndarray, *,
                   time_range: tuple[float, float] | None = None,
                   behaviors: list[BehavioralVT] | None = None,
                   output: PathLike | None = None):
    """
    plot the rastermap result with behavioral measurements

    :param result: :class:`~.core.RasterMapResult`
    :param act_time: neural activity time array. should be the same T as neural_activity when run the rastermap
    :param time_range: time range for plotting (START,END)
    :param behaviors: list of :class:`~BehavioralVT`
    :param output: output path for figure save. If None then show
    """
    plotter = RasterMapPlot(result, act_time, time_range, behaviors, output)
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

    def __init__(self, result: RasterMapResult,
                 act_time: np.ndarray,
                 time_range: tuple[float, float] | None = None,
                 behaviors: list[BehavioralVT] | None = None,
                 output: PathLike | None = None):
        """
        :param result: class:`~.core.RasterMapResult`
        :param act_time: neural activity time array. should be the same T as neural_activity when run the rastermap
        :param time_range: time range for plotting (START,END)
        :param behaviors: list of :class:`~BehavioralVT`
        """
        self.raster = result
        self.behaviors = behaviors

        self.time_range = time_range or (act_time[0], act_time[-1])

        if time_range is not None:
            self.act_mask = np.logical_and(time_range[0] < act_time, act_time < time_range[1])
        else:
            self.act_mask = np.ones_like(act_time, dtype=np.bool_)

        self.act_time = act_time[self.act_mask]

        self.output = output

    @property
    def super_neurons(self) -> np.ndarray:
        """rastermap sorted 2D array. `Array[float, [N, T]]`"""
        return self.raster.super_neurons[:, self.act_mask]

    def process_behavior(self) -> list[BehavioralVT]:
        """process behavioral measurements, select time range and do the interpolation same shape as neural activity"""
        return [
            it.masking_time(self.time_range).interp_activity(self.act_time)
            for it in self.behaviors
        ]

    def plot_rastermap(self):
        if self.behaviors is not None:
            behavior_list = self.process_behavior()
            n_behaviors = len(behavior_list)
        else:
            behavior_list = None
            n_behaviors = 1

        height_ratios = [1] * n_behaviors + [7]

        with plot_figure(self.output, n_behaviors + 1, 1, gridspec_kw={'height_ratios': height_ratios},
                         tight_layout=False, sharex=True) as _ax:

            if self.behaviors is not None:
                for i, it in enumerate(behavior_list):
                    ax = _ax[i]
                    ax.plot(it.time, it.value, color='k')
                    ax.set_xlim(self.time_range)
                    ax.axis('off')
                    ax.set_title(it.name)
            else:
                _ax[0].axis('off')

            ax = _ax[n_behaviors]
            ax.imshow(
                self.super_neurons,
                cmap='gray_r',
                vmin=0,
                vmax=0.8,
                aspect='auto',
                extent=(self.time_range[0], self.time_range[1], self.raster.n_clusters, 0),
            )
            ax.set(xlabel='time', ylabel='rastermap clusters')

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
