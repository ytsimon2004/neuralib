import numpy as np

from .norm import get_percentile_value

__all__ = ['grey2rgb']


def grey2rgb(im: np.ndarray, rgb: int | tuple[int, ...]) -> np.ndarray:
    """
    Converts a grayscale image to an RGB image by assigning intensity values
    to specified color channels.

    :param im: Grey scale image (2d array)
    :param rgb: {0, 1, 2} indicates rgb, tuple represent merge color
    :return: RGB image. `Array[float, [H, W, 3]]`
    """

    if im.ndim != 2:
        raise ValueError('input should be an grayscale 2d array')

    lb, ub = get_percentile_value(im, perc_interval=(10, 100))
    if ub == lb:
        raise ValueError("Cannot normalize with zero range in image.")

    out = np.zeros((*im.shape, 3), float)
    if isinstance(rgb, int):
        out[:, :, rgb] = (im - lb) / ub
    else:
        for i in rgb:
            out[:, :, i] = (im - lb) / ub

    return np.clip(out, 0, 1)
