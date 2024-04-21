from __future__ import annotations

import numpy as np

from neuralib.plot import plot_figure
from neuralib.plot.colormap import get_customized_cmap
from neuralib.util.util_type import ArrayLikeStr, PathLike
from neuralib.wrapper.facemap import FaceMapResult

__all__ = ['plot_facemap_keypoints']


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
