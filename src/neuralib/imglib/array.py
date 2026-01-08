from __future__ import annotations

import cv2
import numpy as np
from typing import Literal, Sequence
from typing import Self

from neuralib.typing import PathLike

__all__ = ['image_array',
           'ImageArrayWrapper']

RGB_CHANNEL_TYPE = Literal['r', 'g', 'b', 'red', 'green', 'blue']


def image_array(dat: np.ndarray | PathLike, *,
                mode: Literal['RGB', 'RGBA', 'gray'] | None = None,
                alpha: bool = False) -> ImageArrayWrapper:
    """
    Image array numpy subclass

    :param dat: Image data as a NumPy array or a file path
    :param mode: Color mode {'RGB', 'RGBA', 'gray'}. Optional if dat is ndarray.
    :param alpha: If True and loading from file, convert to RGBA
    :return:
    """
    return ImageArrayWrapper(dat, mode=mode, alpha=alpha)


class ImageArrayWrapper(np.ndarray):
    """Subclass of numpy.ndarray that wraps an image and provides chainable image processing methods"""

    def __new__(cls, dat: np.ndarray | PathLike, *,
                mode: Literal['RGB', 'RGBA', 'gray'] | None = None,
                alpha: bool = False) -> Self:
        """

        :param dat: Image data as a NumPy array or a file path
        :param mode: Color mode {'RGB', 'RGBA', 'gray'}. Optional if dat is ndarray.
        :param alpha: If True and loading from file, convert to RGBA
        """
        if isinstance(dat, PathLike):
            img = cv2.imread(str(dat))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if alpha:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
                mode = 'RGBA'
            else:
                mode = 'RGB'
            dat = img
        else:
            dat = np.asarray(dat)

            match (mode, dat.ndim, dat.shape[-1] if dat.ndim == 3 else None):
                case (None, 2, _):
                    mode = 'gray'
                case (None, 3, 3):
                    mode = 'RGB'
                case (None, 3, 4):
                    mode = 'RGBA'
                case ('gray', 2, _):
                    pass  # valid grayscale
                case ('RGB', 3, 3):
                    pass  # valid rgb
                case ('RGBA', 3, 4):
                    pass  # valid rgba
                case _:
                    raise ValueError(f"Invalid shape {dat.shape} for mode={mode!r}")

        obj = dat.view(cls)
        obj.mode = mode
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.mode = getattr(obj, 'mode', 'RGB')

    @property
    def height(self) -> int:
        """image height"""
        return self.shape[0]

    @property
    def width(self) -> int:
        """image width"""
        return self.shape[1]

    def to_gray(self) -> Self:
        """convert the image array to grayscale"""
        if self.ndim == 2:
            ret = self.copy().astype("uint8")
        elif self.shape[2] == 3:
            ret = cv2.cvtColor(self, cv2.COLOR_RGB2GRAY)
        elif self.shape[2] == 4:
            ret = cv2.cvtColor(self, cv2.COLOR_RGBA2GRAY)
        else:
            raise RuntimeError(f'Unexpected image shape: {self.shape} for grayscale conversion')

        return ImageArrayWrapper(ret, mode='gray')

    def flipud(self) -> Self:
        """flip the image array upside down (vertical)"""
        return ImageArrayWrapper(np.flipud(self))

    def fliplr(self) -> Self:
        """flip the image array left to right (horizontal flip)"""
        return ImageArrayWrapper(np.fliplr(self))

    def select_channel(self, channel: RGB_CHANNEL_TYPE) -> Self:
        """extract a single color channel from an RGB or RGBA image

        :param channel: one of 'r'/'red', 'g'/'green', or 'b'/'blue'.
        """
        if self.ndim < 3:
            raise RuntimeError(f'Image shape invalid for splitting channel: {self.ndim}')

        channels = cv2.split(self)
        match channel:
            case 'r' | 'red':
                ret = channels[0]
            case 'g' | 'green':
                ret = channels[1]
            case 'b' | 'blue':
                ret = channels[2]
            case _:
                raise ValueError(f'invalid {channel} argument')

        return ImageArrayWrapper(ret, mode='gray')

    def view_2d(self, flipud: bool = False) -> Self:
        """
        Convert a multi-channel image to a 2D representation.

        - For a 4-channel image, the array is reinterpreted as a 2D array of 32-bit integers.
        - For a 3-channel image, the result is a grayscale image obtained by applying luminance conversion.

        :param flipud: reverse the order of elements along axis 0 (up/down)
        """
        if self.ndim == 2:
            return self

        match self.shape[2]:
            case 4:
                w, h, _ = self.shape
                ret = self.view(dtype=np.uint32).reshape((w, h))
            case 3:
                r, g, b = self[:, :, 0], self[:, :, 1], self[:, :, 2]
                ret = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.uint8)
            case _:
                raise ValueError(f'invalid arr shape: {self.shape}')

        zelf = ImageArrayWrapper(ret, mode='gray')

        if flipud:
            zelf = zelf.flipud()

        return zelf

    def gaussian_blur(self, ksize: Sequence[int], sigma_x: float, sigma_y: float, **kwargs) -> Self:
        """
        Apply a Gaussian blur to the image.

        :param ksize: Kernel size (e.g., (5, 5)). The width and height should be odd numbers.
        :param sigma_x: Standard deviation in the X direction.
        :param sigma_y: Standard deviation in the Y direction.
        :param kwargs: Additional keyword arguments for ``cv2.GaussianBlur()``.
        """
        img = cv2.GaussianBlur(self, ksize=ksize, sigmaX=sigma_x, sigmaY=sigma_y, **kwargs)
        return ImageArrayWrapper(img)

    def canny_filter(self, threshold_1: float = 30, threshold_2: float = 150, **kwargs) -> Self:
        """
        Apply the Canny edge detection algorithm to the grayscale version of the image.

        :param threshold_1: The first threshold for the hysteresis procedure.
        :param threshold_2: The second threshold for the hysteresis procedure.
        :param kwargs: Additional keyword arguments for ``cv2.Canny()``.
        """
        img = cv2.Canny(self.to_gray(), threshold1=threshold_1, threshold2=threshold_2, **kwargs)
        return ImageArrayWrapper(img)

    def binarize(self, thresh: float, maxval: float = 255, **kwargs) -> Self:
        """
        Convert the image to a binary image using a fixed threshold.

        :param thresh: Threshold value. Pixels above this value are set to maxval; otherwise, 0.
        :param maxval: The value to use for pixels above the threshold.
        :param kwargs: Additional keyword arguments for ``cv2.threshold()``.
        """
        _, img = cv2.threshold(self, thresh, maxval=maxval, type=cv2.THRESH_BINARY, **kwargs)
        return ImageArrayWrapper(img)

    def denoise(self, h: int = 10, temp_win_size: int = 7, search_win_size: int = 21, **kwargs) -> Self:
        """
        Apply Non-local Means Denoising to the image.

        - For grayscale images, cv2.fastNlMeansDenoising is used.
        - For color images, cv2.fastNlMeansDenoisingColored is used.

        :param h: Filtering parameter controlling the degree of smoothing.
        :param temp_win_size: Template window size in pixels.
        :param search_win_size: Search window size in pixels.
        :param kwargs: Additional keyword arguments for the cv2 denoising function.
        """
        if self.mode == 'gray':
            fn = getattr(cv2, 'fastNlMeansDenoising')
        else:
            fn = getattr(cv2, 'fastNlMeansDenoisingColored')
        img = fn(self, h=h, templateWindowSize=temp_win_size, searchWindowSize=search_win_size, **kwargs)
        return ImageArrayWrapper(img)

    def enhance_contrast(self) -> Self:
        """Enhance the contrast of the image using histogram equalization"""
        match self.mode:
            case 'gray':
                eq = cv2.equalizeHist(self)
                return ImageArrayWrapper(eq, mode='gray')
            case 'RGB':
                ycrcb = cv2.cvtColor(self, cv2.COLOR_RGB2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                eq = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
                return ImageArrayWrapper(eq, mode='RGB')
            case 'RGBA':
                rgb = self[..., :3]
                alpha = self[..., 3]
                ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                eq_rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
                eq = np.dstack((eq_rgb, alpha))
                return ImageArrayWrapper(eq, mode='RGBA')
            case _:
                raise ValueError(f'invalid mode: {self.mode}')

    def local_maxima(self, channel: RGB_CHANNEL_TYPE, **kwargs) -> Self:
        """
        Compute the local maxima of the image on a specified color channel.

        The specified channel is first extracted (and returned as a grayscale image),
        then the skimage local_maxima function is applied.
        :param channel: one of 'r'/'red', 'g'/'green', or 'b'/'blue'.
        :param kwargs: additional keyword arguments for ``skimage.morphology.local_maxima()``.
        """
        from skimage.morphology import local_maxima

        img = self.select_channel(channel)

        if np.sum(img) == 0:
            return ImageArrayWrapper(np.zeros_like(img, dtype=np.uint8), mode='gray')
        else:
            return ImageArrayWrapper(local_maxima(img, **kwargs), mode='gray')
