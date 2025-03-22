

import cv2
import numpy as np

__all__ = ['affine_transform',
           'apply_transformation']


def affine_transform(src: np.ndarray) -> np.ndarray:
    """
    Do affine transformation on flatten image

    .. seealso::

        https://docs.opencv.org/4.x/d4/d61/tutorial_warp_affine.html

    :param src: source image array
    :return: output image array
    """
    srcTri = np.array([
        [0, 0],
        [src.shape[1] - 1, 0],
        [0, src.shape[0] - 1]
    ]).astype(np.float32)

    dstTri = np.array([
        [0, src.shape[1] * 0.33],
        [src.shape[1] * 0.6, src.shape[0] * 0.25],
        [src.shape[1] * 0.4, src.shape[0] * 0.7]
    ]).astype(np.float32)

    mat = cv2.getAffineTransform(srcTri, dstTri)

    warp_dst = cv2.warpAffine(src, mat, (src.shape[1], src.shape[0]))

    return warp_dst


def apply_transformation(img: np.ndarray,
                         trans_mtx: np.ndarray,
                         **kwargs) -> np.ndarray:
    """
    2D image transform

    :param img: image array
    :param trans_mtx: Transformation matrix. `Array[float, [3, 3]]`
    :param kwargs: additional arguments pass to ``cv2.warpPerspective()``
    :return:
    """
    height, width, _ = img.shape

    if trans_mtx.shape != (3, 3):
        raise ValueError(f'invalid transformation shape: {trans_mtx.shape}')

    return cv2.warpPerspective(img, trans_mtx, (width, height), **kwargs)
