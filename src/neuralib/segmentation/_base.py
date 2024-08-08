import abc
from pathlib import Path

import cv2
import numpy as np

from neuralib.argp import AbstractParser, argument

__all__ = ['AbstractSegmentationOption']


class AbstractSegmentationOption(AbstractParser, metaclass=abc.ABCMeta):
    DESCRIPTION = 'ABC for cellular segmentation'

    file: Path = argument('-F', '--file', help='image file path')
    directory: Path = argument('-D', '--dir', help='images directory for batch processing')
    model: str = argument('-M', '--model', metavar='MODEL', help='which pretrained model')
    napari_view: bool = argument('--napari', help='view in napari')

    no_normalize: bool = argument('--no-norm', help='NOT DO Percentile-based image normalization for eval')
    force_re_eval: bool = argument('--force-eval', '--re', help='force re-evaluate the result')

    def post_parsing(self):
        if not self.file and not self.directory:
            raise ValueError('Either file or directory must be specified for processing')

        if self.file and self.directory:
            raise ValueError('Only one of file or directory should be specified, not both')

    @property
    def file_mode(self) -> bool:
        return self.file is not None

    @property
    def batch_mode(self) -> bool:
        return self.directory is not None

    def raw_image(self) -> np.ndarray:
        """load image from file and convert to grayscale

        :return: `Array[float, [H, W]]`
        """
        if not self.file.is_file():
            raise ValueError('not a file')

        img = cv2.imread(str(self.file))
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img

    def normalize_image(self) -> np.ndarray:
        """
        Normalize the image

        :return: `Array[float, [H, W]]`
        """
        from csbdeep.utils import normalize
        img = self.raw_image()

        return normalize(img, clip=True)

    @abc.abstractmethod
    def eval(self) -> None:
        """eval the model, and save the results"""
        pass

    @abc.abstractmethod
    def launch_napari(self, **kwargs):
        """napari viewer"""
        pass
