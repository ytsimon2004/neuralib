import cv2
import numpy as np

__all__ = ['affine_transform']


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
