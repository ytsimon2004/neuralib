from __future__ import annotations

from typing import final, Literal

import attrs
import cv2
import numpy as np
from typing_extensions import Self

from neuralib.typing import PathLike

__all__ = [
    'ImageProcFactory',
    'IMAGE_CHANNEL_TYPE',
    #
    'recover_overexposure',
]

IMAGE_CHANNEL_TYPE = Literal['red', 'green', 'blue', 'r', 'g', 'b']


@final
@attrs.define(repr=False)
class ImageProcFactory:
    """Factory for basic imaging processing"""
    image: np.ndarray
    """image array. `Array[float, [H, W]|[H, W, 3]|[H, W, 4]]`"""

    @classmethod
    def load(cls, file: PathLike,
             to_rgba: bool | None = None,
             to_rgb: bool | None = True) -> Self:
        """
        Load the image file

        :param file: filepath of the image
        :param to_rgba: convert to RGBA colorscale
        :param to_rgb: convert to RGB colorscale
        :return:
        """

        img = cv2.imread(str(file))

        if to_rgba:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
            img = np.array(img, dtype=np.uint8)

        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return ImageProcFactory(img)

    @property
    def height(self) -> int:
        """image height `H`"""
        return self.image.shape[0]

    @property
    def width(self) -> int:
        """image width `W`"""
        return self.image.shape[1]

    def select_channel(self, channel: IMAGE_CHANNEL_TYPE) -> Self:
        """Select `RGB` channel

        :param channel: {red', 'green', 'blue', 'r', 'g', 'b'}
        """

        if self.image.ndim < 3:
            raise TypeError('shape invalid for splitting channel')

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
        """view `RGB` or `RGBA` image as 2d array. `Array[float, [H, W]]`"""
        if self.image.shape[2] == 4:
            w, h, _ = self.image.shape
            img = self.image.view(dtype=np.uint32).reshape((w, h))
        elif self.image.shape[2] == 3:
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

    def cvt_gray(self) -> Self:
        """convert to grayscale"""
        if self.image.ndim == 2:
            return attrs.evolve(self, image=np.uint8(self.image))
        elif self.image.shape[2] == 3:
            return attrs.evolve(self, image=cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY))
        elif self.image.shape[2] == 4:
            return attrs.evolve(self, image=(cv2.cvtColor(self.image, cv2.COLOR_RGBA2GRAY)))
        else:
            raise RuntimeError('')

    def gaussian_blur(self, ksize: int, sigma: int) -> Self:
        img = cv2.GaussianBlur(self.image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        return attrs.evolve(self, image=img)

    def edge_detection(self,
                       lower_threshold: int = 30,
                       upper_threshold: int = 150) -> Self:
        grey_img = self.cvt_gray().image
        img = cv2.Canny(grey_img, lower_threshold, upper_threshold)
        return attrs.evolve(self, image=img)

    def binarize(self, threshold: int = 150) -> Self:
        _, img = cv2.threshold(self.image, threshold, 255, cv2.THRESH_BINARY)
        return attrs.evolve(self, image=img)

    def de_noise(self, h: int = 10, temp_win_size: int = 7, search_win_size: int = 21) -> Self:
        gray_img = self.cvt_gray().image
        dn = cv2.fastNlMeansDenoising(gray_img,
                                      h=h,
                                      templateWindowSize=temp_win_size,
                                      searchWindowSize=search_win_size)
        return attrs.evolve(self, image=dn)

    def local_maxima_image(self, channel: IMAGE_CHANNEL_TYPE, **kwargs) -> Self:
        """
        find the local maxima of the selection points.
        i.e., used in roi selection of the neuron before counting

        :param channel: color of image
        :return: `Array[float, [H, W]]`
        """
        from skimage.morphology import local_maxima

        image = self.select_channel(channel).image
        if np.sum(image) == 0:
            return attrs.evolve(self, image=np.zeros_like(image, dtype=np.uint8))
        else:
            return attrs.evolve(self, image=local_maxima(image, **kwargs))


def recover_overexposure(img: np.ndarray,
                         alpha: float = 0.5,
                         beta: float = 0.5) -> np.ndarray:
    """recover saturated fluorescence image"""
    proc = cv2.addWeighted(img, alpha, np.zeros_like(img), beta, 0.0)
    return proc
