
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from neuralib.plot import plot_figure
from neuralib.plot.colormap import get_customized_cmap
from neuralib.tracking.facemap import FaceMapResult
from neuralib.typing import ArrayLikeStr, PathLike

__all__ = ['plot_facemap_keypoints',
           'plot_cmap_time_series']


def plot_facemap_keypoints(fmap: FaceMapResult,
                           frame_interval: tuple[int, int],
                           keypoints: str | ArrayLikeStr | None = None,
                           outlier_filter: bool = True,
                           output: PathLike | None = None) -> None:
    """
    Plot all the keypoints

    :param fmap: :class:`~neuralib.wrapper.facemap.core.FaceMapResult`
    :param frame_interval: frame interval (start/end) for the plot
    :param keypoints: a keypoint name or multiple keypoints. If None, then show all the keypoints
    :param outlier_filter: remove jump and do the interpolation
    :param output: output file. Show the fig if None
    """
    if keypoints is None:
        kps = fmap.keypoints
    else:
        kps = list(keypoints)

    x = np.arange(*frame_interval)
    colors = get_customized_cmap('jet', (0, 1), len(kps))

    with plot_figure(output) as ax:
        for i, kp in enumerate(fmap.get(kps)):

            if outlier_filter:
                _kp = kp.with_outlier_filter()
            else:
                _kp = kp

            ax.plot(x, _kp.x[x], '-', color=colors[i], label=f'{_kp.name}_x')
            ax.plot(x, _kp.y[x], '--', color=colors[i], label=f'{_kp.name}_y')

        ax.set(xlabel='frame', ylabel='keypoint coordinates')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)


def plot_cmap_time_series(x: np.ndarray,
                          y: np.ndarray, *,
                          cmap: str = 'viridis',
                          ax: Axes | None = None,
                          with_color_bar: bool = True,
                          color_bar_label: str = 'frames',
                          output: PathLike | None = None,
                          **kwargs) -> None:
    """
    Plots a scatter plot with a colorbar

    :param x: X-axis values. `Array[float, T]`
    :param y: Y-axis values. `Array[float, T]`
    :param cmap: Colormap to use
    :param ax: ``matplotlib.axes.Axes``
    :param with_color_bar: if show the colorbar
    :param color_bar_label: color bar label
    :param output: Output file path to save the plot
    :param kwargs: pass through ``ax.scatter()``
    """
    if x.size != y.size:
        raise ValueError('size xy inconsistent')

    colors = get_customized_cmap(cmap, value=(0, 1), numbers=len(x))
    norm = Normalize(vmin=0, vmax=len(x))

    def _plot(ax: Axes):
        ax.scatter(x, y, c=colors, alpha=0.5, **kwargs)
        if with_color_bar:
            cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
            cbar.set_label(color_bar_label)
            ax.set(xlabel='x', ylabel='y')

    #
    if ax is None:
        with plot_figure(output) as ax:
            _plot(ax)
    else:
        _plot(ax)
        if output is not None:
            plt.savefig(output)
