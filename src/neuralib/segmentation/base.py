import abc
from pathlib import Path
from typing import Iterable, Literal

import cv2
import numpy as np

from argclz import AbstractParser, argument

__all__ = ['AbstractSegmentationOptions']


class AbstractSegmentationOptions(AbstractParser, metaclass=abc.ABCMeta):
    DESCRIPTION = 'Base Cellular Segmentation Option'

    # ---- GROUP_IO -----------------------------
    GROUP_IO = 'Data I/O Options'
    EX_GROUP_SOURCE = 'EX_GROUP_SOURCE'

    file: Path = argument(
        '--file', '--image_path',
        ex_group=EX_GROUP_SOURCE,
        group=GROUP_IO,
        help='image file path'
    )

    directory: Path = argument(
        '--dir',
        ex_group=EX_GROUP_SOURCE,
        group=GROUP_IO,
        help='directory for batch imaging processing'
    )

    directory_suffix: Literal['.tif', '.tiff', '.png'] = argument(
        '--dir_suffix',
        default='.tif',
        group=GROUP_IO,
        help='suffix in the directory for batch mode'
    )

    save_ij_roi: bool = argument(
        '--save_rois',
        group=GROUP_IO,
        help='if save also the imageJ/Fiji compatible .roi file'
    )

    # ---- OTHERS -----------------------------
    model: str = argument(
        '--model',
        help='which pretrained model for evaluation'
    )

    invalid_existed_result: bool = argument(
        '--invalid',
        help='force re-evaluate the result'
    )

    no_normalize: bool = argument(
        '--no_norm',
        help='not do percentile-based image normalization'
    )

    napari_view: bool = argument(
        '--napari',
        help='view result by napari GUI, only available in single file mode'
    )

    @property
    def file_mode(self) -> bool:
        """flag file mode"""
        return self.file is not None

    @property
    def batch_mode(self) -> bool:
        """flag batch mode"""
        return self.directory is not None

    @property
    def with_norm(self) -> bool:
        """flag normalize image"""
        return not self.no_normalize

    def process_image(self, to_gray: bool = True) -> np.ndarray:
        """Process the image for segmentation.

        :return: `Array[Any, [H, W]]` or `Array[Any, [H, W, C]]`
        """
        return process_image(
            cv2.imread(str(self.file)),
            to_gray=to_gray,
            norm=self.with_norm
        )

    def foreach_process_image(self, to_gray: bool = True) -> Iterable[tuple[Path, np.ndarray]]:
        """
        Iterates over image files in the specified directory, processes each image, and yields
        the file path along with the processed image. The processing can include grayscale
        conversion and normalization based on the provided parameters.

        :param to_gray: Flag indicating whether the images should be converted to grayscale.
        :return: An iterable of tuples where each tuple includes the file path as a `Path` object
            and the processed image as a numpy array.
        :rtype: Tuple of filepath, image_array (`Array[Any, [H, W]]` or `Array[Any, [H, W, C]]`) generator
        """
        for file in self.directory.glob(f'*{self.directory_suffix}'):
            img = process_image(
                cv2.imread(str(file)),
                to_gray=to_gray,
                norm=self.with_norm
            )
            yield file, img

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
        """eval the model in single file or batch files in directory, and save the results"""
        pass

    @abc.abstractmethod
    def launch_napari(self, **kwargs):
        """run napari GUI viewer"""
        pass


def process_image(img: np.ndarray,
                  to_gray: bool = True,
                  norm: bool = True) -> np.ndarray:
    """
    Pre process the image for segmentation.

    :param img: image array, `Array[Any, [H, W]]` or `Array[Any, [H, W, C]]`
    :param to_gray: to grayscale. default to True
    :param norm: to min-max normalize, default to True
    :return: pre-processed image array, `Array[Any, [H, W]]` or `Array[Any, [H, W, C]]`
    """
    if to_gray and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if norm:
        from csbdeep.utils import normalize
        img = normalize(img, clip=True, pmin=20)

    return img
