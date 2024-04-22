from __future__ import annotations

import numpy as np

__all__ = ['grey2rgb']


def grey2rgb(im: np.ndarray,
             rgb: int | tuple[int, ...], *,
             out: np.ndarray | None = None):
    """
    Transfer grey scale image to rgb

    :param im: grey scale image (2d array)
    :param rgb: {0, 1, 2} indicates rgb, tuple represent merge color
    :param out: output array
    :return:
    """
    from neuralib.imglib.norm import get_percentile_value

    if im.ndim != 2:
        raise ValueError('input should be an grayscale 2d array')

    #
    if out is None:
        out = np.zeros((*im.shape, 3), float)
    else:
        if out.shape != (*im.shape, 3):
            raise ValueError

    #
    lb, ub = get_percentile_value(im, perc_interval=(10, 100))
    if ub == lb:
        raise ValueError("Cannot normalize with zero range in image.")

    #
    if isinstance(rgb, int):
        out[:, :, rgb] = (im - lb) / ub
    else:
        for i in rgb:
            out[:, :, i] = (im - lb) / ub

    return np.clip(out, 0, 1)
