from __future__ import annotations

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from neuralib.plot import plot_figure
from neuralib.plot.colormap import get_customized_cmap
from neuralib.util.util_type import ArrayLikeStr, PathLike
from neuralib.wrapper.facemap import FaceMapResult

__all__ = ['plot_facemap_keypoints',
           'plot_camp_time_series']


def plot_facemap_keypoints(fmap: FaceMapResult,
                           frame_interval: tuple[int, int],
                           keypoints: str | ArrayLikeStr | None = None,
                           outlier_filter: bool = True,
                           output: PathLike | None = None):
    """
    Plot all the keypoints

    :param fmap: :class:`~neuralib.wrapper.facemap.core.FaceMapResult`
    :param frame_interval: frame interval (start/end) for the plot
    :param keypoints: a keypoint name Or multiple keypoints. If None, then show all the keypoints
    :param outlier_filter: remove jump and do the interpolation
    :param output: output file. Show the fig if None
    :return:
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


def plot_camp_time_series(x: np.ndarray,
                          y: np.ndarray,
                          cmap: str = 'viridis',
                          output: PathLike | None = None,
                          **kwargs):
    """
    Plots a scatter plot with a colorbar

    :param x: X-axis values
    :param y: Y-axis values
    :param cmap: Colormap to use
    :param output: Output file path to save the plot
    :return:
    """
    if x.size != y.size:
        raise ValueError('size xy inconsistent')

    colors = get_customized_cmap(cmap, value=(0, 1), numbers=len(x))
    norm = Normalize(vmin=0, vmax=len(x))

    with plot_figure(output) as ax:
        ax.scatter(x, y, c=colors, alpha=0.5, **kwargs)
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cbar.set_label('frames')
        ax.set(xlabel='x', ylabel='y')
