from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes

from neuralib.calimg.suite2p import Suite2PResult

__all__ = ['get_soma_pixel',
           'plot_soma_center']


def get_soma_pixel(s2p: Suite2PResult,
                   neuron_ids: np.ndarray | None = None, *,
                   color_diff: bool = True,
                   include_overlap_pixel: bool = True) -> np.ndarray:
    """
    Get the image mask of the registered neuronal soma shape.
    The mask value equals to the neuron's id + 1.

    :param s2p: :class:`~neuralib.calimg.suite2p.core.Suite2PResult`
    :param neuron_ids: index ROIs array or bool mask array
    :param color_diff: whether show color difference across neurons
    :param include_overlap_pixel: if taking overlap area into account
    :return: 2d imaging array, shape: (xpix, ypix)
    """
    neuron_pix = np.zeros((s2p.image_width, s2p.image_height))

    if neuron_ids is None:
        neuron_ids = np.arange(s2p.n_neurons)

    elif neuron_ids.dtype == bool:
        if len(neuron_ids) != s2p.stat.shape[0]:
            raise ValueError('invalid shape for the mask array')

        neuron_ids = np.nonzero(neuron_ids)[0]

    for i in neuron_ids:
        ypix = s2p.stat[i]['ypix']
        xpix = s2p.stat[i]['xpix']

        if not include_overlap_pixel:
            ypix = ypix[~s2p.stat[i]['overlap']]
            xpix = xpix[~s2p.stat[i]['overlap']]

        neuron_pix[xpix, ypix] = i + 1 if color_diff else 1

    return neuron_pix


def plot_soma_center(ax: Axes,
                     s2p: Suite2PResult,
                     neuron_ids: np.ndarray | None = None, *,
                     invert_xy: bool = True,
                     with_index: bool = True,
                     font_size: float = 5,
                     **kwargs) -> None:
    """Plot center of roi soma and its corresponding index

    :param ax: :class:`matplotlib.axes.Axes`
    :param s2p: :class:`~neuralib.calimg.suite2p.core.Suite2PResult`
    :param neuron_ids: index ROIs array or bool mask array
    :param invert_xy: If invert the FOV xy pixel
    :param with_index: whether plot the index ROI index nearby
    :param font_size: font size of the text
    :param kwargs: pass to plt.scatter()
    """
    if neuron_ids is None:
        n_neuron = s2p.stat.shape[0]
        neuron_ids = np.arange(n_neuron)

    elif neuron_ids.dtype == bool:
        if len(neuron_ids) != s2p.stat.shape[0]:
            raise ValueError('invalid shape for the mask array')
        neuron_ids = np.nonzero(neuron_ids)[0]

    coords = np.array([(np.mean(s2p.stat[i]['ypix' if invert_xy else 'xpix']),
                        np.mean(s2p.stat[i]['xpix' if invert_xy else 'ypix'])) for i in neuron_ids])

    ax.scatter(coords[:, 0], coords[:, 1], **kwargs)

    if with_index:
        for (x, y), idx in zip(coords, neuron_ids):
            ax.text(x, y, str(idx), fontsize=font_size)
