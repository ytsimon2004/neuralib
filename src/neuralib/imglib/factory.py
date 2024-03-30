from __future__ import annotations

from typing import Generic, TypeVar, final, Literal

import attrs
import cv2
import numpy as np
from typing_extensions import Self

from neuralib.util.util_type import PathLike

__all__ = [
    'ImageProcFactory',
    'IMAGE_CHANNEL_TYPE',
    #
    'apply_transformation',
    'detect_feature'
]

T = TypeVar('T')
IMAGE_CHANNEL_TYPE = Literal['red', 'green', 'blue', 'r', 'g', 'b']


@final
@attrs.define(repr=False)
class ImageProcFactory(Generic[T]):
    """Factory for basic imaging processing"""

    image: np.ndarray
    """(H, W, 4 | 3) or (H, W)"""

    color_type: T | None = attrs.field(init=False, default=None)
    """image color type"""

    def __repr__(self):
        shape = self.image.shape
        ctype = self.color_type
        return f'Shape: {shape}, color: {ctype}'

    def __attrs_post_init__(self):
        if self.image.ndim == 2:
            self.color_type = 'GrayScale'
        elif self.image.ndim == 3:
            c = self.image.shape[2]
            if c == 3:
                self.color_type = 'RGB'
            elif c == 4:
                self.color_type = 'RGBA'

        assert self.color_type is not None, 'color type could not be inferred'

    @classmethod
    def load(cls, file: PathLike,
             to_rgba: bool | None = None,
             to_rgb: bool | None = True) -> Self:

        img = cv2.imread(str(file))

        if to_rgba:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            img = np.array(img, dtype=np.uint8)

        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return ImageProcFactory(img)

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]

    def with_channel(self, channel: IMAGE_CHANNEL_TYPE) -> Self:
        """select rgb channel"""

        if self.color_type not in ('RGBA', 'RGB'):
            raise TypeError(f'{self.color_type} not able to split channel')

        if channel in ('r', 'red'):
            img = cv2.split(self.image)[0]
        elif channel in ('g', 'green'):
            img = cv2.split(self.image)[1]
        elif channel in ('b', 'blue'):
            img = cv2.split(self.image)[2]
        else:
            raise KeyError(f'{channel}')

        return attrs.evolve(self, image=img)

    def view_2d(self, flip: bool = True) -> Self:
        """view image as 2d array (H, W)"""
        if self.color_type == 'RGBA':
            w, h, _ = self.image.shape
            img = self.image.view(dtype=np.uint32).reshape((w, h))
        elif self.color_type == 'RGB':
            r = self.image[:, :, 0]
            g = self.image[:, :, 1]
            b = self.image[:, :, 2]
            grayscale_image = 0.2989 * r + 0.5870 * g + 0.1140 * b
            img = grayscale_image.astype(np.uint8)
        else:
            raise RuntimeError(f'{self.image.shape} check')

        if flip:
            img = np.flipud(img)

        return attrs.evolve(self, image=img)

    # ============= #
    # Basic Process #
    # ============= #
    def covert_grey_scale(self) -> Self:
        if self.color_type == 'GrayScale':
            return attrs.evolve(self, image=np.uint8(self.image))
        else:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            return attrs.evolve(self, image=img)

    def gaussian_blur(self, ksize: int, sigma: int) -> Self:
        img = cv2.GaussianBlur(self.image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        return attrs.evolve(self, image=img)

    def edge_detection(self,
                       lower_threshold: int = 30,
                       upper_threshold: int = 150) -> Self:
        grey_img = self.covert_grey_scale().image
        img = cv2.Canny(grey_img, lower_threshold, upper_threshold)
        return attrs.evolve(self, image=img)

    def binarize(self, threshold: int = 150) -> Self:
        _, img = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)
        return attrs.evolve(self, image=img)

    def de_noise(self) -> Self:
        gray_img = self.covert_grey_scale().image
        return attrs.evolve(self, image=de_noise(gray_img))

    def local_maxima_image(self, channel: IMAGE_CHANNEL_TYPE, **kwargs) -> Self:
        """
        find the local maxima of the selection points.
        i.e., used in roi selection of the neuron before counting

        :param channel: color of image
        :return:
            (H, W) 2d array
        """
        from skimage.morphology import local_maxima

        image = self.with_channel(channel).image
        if np.sum(image) == 0:
            return attrs.evolve(self, image=np.zeros_like(image, dtype=np.uint8))
        else:
            return attrs.evolve(self, image=local_maxima(image, **kwargs))


def de_noise(gray_img: np.ndarray) -> np.ndarray:
    # https://stackoverflow.com/questions/62042172/how-to-remove-noise-in-image-opencv-python
    blur = cv2.GaussianBlur(gray_img, (3, 3), sigmaX=30, sigmaY=30)
    divide = cv2.divide(gray_img, blur, scale=255)
    # otsu threshold
    thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # apply morphology
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    cv2.imwrite("test.jpg", thresh)

    return thresh


def recover_overexposure(img: np.ndarray,
                         alpha: float = 0.5,
                         beta: float = 0.5) -> np.ndarray:
    """recover saturated fluorescence image"""
    proc = cv2.addWeighted(img, alpha, np.zeros_like(img), beta, 0.0)
    return proc


def apply_transformation(img: np.ndarray,
                         trans_mtx: np.ndarray,
                         **kwargs) -> np.ndarray:
    """2D image transform"""

    height, width, _ = img.shape

    if trans_mtx.shape != (3, 3):
        raise ValueError(f'invalid transformation shape: {trans_mtx.shape}')

    return cv2.warpPerspective(img, trans_mtx, (width, height), **kwargs)


def detect_feature(img: np.ndarray,
                   filter_type: str = 'ORB') -> tuple[np.ndarray, np.ndarray]:
    if filter_type == 'ORB':
        f = cv2.ORB_create()
    elif filter_type == 'SIFT':
        f = cv2.SIFT_create()
    else:
        raise NotImplementedError('')
    keypoints, descriptors = f.detectAndCompute(img, None)

    return keypoints, descriptors
