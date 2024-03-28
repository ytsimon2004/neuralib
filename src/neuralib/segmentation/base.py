import abc
from pathlib import Path

import cv2
import numpy as np

from neuralib.argp import AbstractParser, argument

__all__ = ['AbstractSegmentationOption']


class AbstractSegmentationOption(AbstractParser, metaclass=abc.ABCMeta):
    DESCRIPTION = 'ABC for cellular segmentation'

    file: Path = argument('-F', '--file', required=True, help='image file path')
    model: str = argument('-M', '--model', help='which pretrained model')
    napari_view: bool = argument('--napari', help='view in napari')

    no_normalize: bool = argument('--no-norm', help='NOT DO Percentile-based image normalization for eval')

    def load_gray_scale(self, normalize: bool = True) -> np.ndarray:
        if not self.file.is_file():
            raise ValueError('not a file')

        img = cv2.imread(str(self.file))
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if normalize:
            from csbdeep.utils import normalize
            img = normalize(img, clip=True)

        return img

    def get_raw_image(self) -> np.ndarray:
        pass

    @abc.abstractmethod
    def eval(self):
        """eval the model"""
        pass

    @abc.abstractmethod
    def launch_napari(self, **kwargs):
        """napari viewer"""
        pass
