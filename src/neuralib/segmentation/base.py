import abc
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from csbdeep.utils import normalize

from neuralib.argp import AbstractParser, argument

__all__ = ['AbstractSegmentationOption']


class AbstractSegmentationOption(AbstractParser, metaclass=abc.ABCMeta):
    DESCRIPTION = 'ABC for cellular segmentation'

    EX_GROUP_SOURCE = 'EX_GROUP_SOURCE'

    file: Path = argument(
        '-F', '--file',
        required=True,
        ex_group=EX_GROUP_SOURCE,
        help='image file path'
    )

    directory: Path = argument(
        '-D', '--dir',
        ex_group=EX_GROUP_SOURCE,
        help='images directory for batch processing'
    )

    directory_suffix: str = argument(
        '--suffix',
        default='.tif',
        choices=['.tif', '.tiff', '.png'],
        help='suffix in batch mode'
    )

    save_ij_roi: bool = argument(
        '--ij-roi',
        '--roi',
        help='if save also the imageJ/Fiji compatible .roi file'
    )

    force_re_eval: bool = argument(
        '--force-eval', '--re',
        help='force re-evaluate the result'
    )

    model: str = argument(
        '-M', '--model', metavar='MODEL',
        help='which pretrained model'
    )

    no_normalize: bool = argument(
        '--no-norm',
        help='NOT DO Percentile-based image normalization for eval'
    )

    napari_view: bool = argument(
        '--napari',
        help='view in napari'
    )

    def post_parsing(self):
        """check args is valid"""
        if self.directory and self.napari_view:
            raise ValueError('napari view only used in single file mode')

    @property
    def file_mode(self) -> bool:
        """Flag file mode"""
        return self.file is not None

    @property
    def batch_mode(self) -> bool:
        """Flag batch mode"""
        return self.directory is not None

    @staticmethod
    def _as_grayscale(file: str) -> np.ndarray:
        img = cv2.imread(file)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def raw_image(self) -> np.ndarray:
        """Load image from file and convert to grayscale

        :return: `Array[float, [H, W]]`
        """
        if not self.file.is_file():
            raise ValueError('not a file')

        return self._as_grayscale(str(self.file))

    def normalize_image(self) -> np.ndarray:
        """
        Normalize the image

        :return: `Array[float, [H, W]]`
        """
        img = self.raw_image()

        return normalize(img, clip=True)

    def foreach_raw_image(self) -> Iterable[tuple[Path, np.ndarray]]:
        """
        Load image from a directory and convert to grayscale

        :return: Tuple of filepath and image `Array[float, [H, W]]`
        """
        for file in self.directory.glob(f'*{self.directory_suffix}'):
            img = self._as_grayscale(str(file))
            yield file, img

    def foreach_normalize_image(self) -> Iterable[tuple[Path, np.ndarray]]:
        """
        Normalize the image in batch mode

        :return: Tuple of filepath and image `Array[float, [H, W]]`
        """
        for name, raw in self.foreach_raw_image():
            yield name, normalize(raw, clip=True)

    @abc.abstractmethod
    def seg_output(self, filepath: Path) -> Path:
        """
        Get segmented output save path

        :param filepath: filepath for image
        :return: segmented output save path
        """
        pass

    def ij_roi_output(self, filepath: Path) -> Path:
        """
        Get imageJ/Fiji ``.roi`` output save path

        :param filepath: filepath for image
        :return: ij roi output save path
        """
        return filepath.with_suffix('.roi')

    @abc.abstractmethod
    def eval(self) -> None:
        """eval the model in single file or batch files, and save the results"""
        pass

    @abc.abstractmethod
    def launch_napari(self, **kwargs):
        """napari viewer"""
        pass
