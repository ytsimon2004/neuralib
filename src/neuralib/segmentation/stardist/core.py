from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.lib.npyio import NpzFile
from typing import Self, Literal

from neuralib.typing import PathLike

__all__ = [
    'STARDIST_MODEL',
    'read_stardist',
    'stardist_point_roi_helper',
    'StarDistSegmentation'
]

STARDIST_MODEL = Literal['2D_versatile_fluo', '2D_versatile_he', '2D_paper_dsb2018', '2D_demo']
"""stardist supported model type"""


def read_stardist(file: PathLike) -> StarDistSegmentation:
    """
    Read a cellpose segmentation result file

    :param file: stardist segmentation result ``.npy`` file
    :return: :class:`StarDistSegmentation`
    """
    return StarDistSegmentation.load(file)


def stardist_point_roi_helper(file: PathLike, output: PathLike) -> None:
    """Read a stardist segmentation result and convert the segmentation result to point coordinates

    :param file: stardist segmentation result ``.npz`` file
    :param output: ``*.roi`` output file path
    """
    StarDistSegmentation.load(file).to_roi(output)


class StarDistSegmentation:
    """`StarDist <https://github.com/stardist/stardist>`_ segmentation results

    `Dimension parameters`:

        N = Number of segmented cell

        E = Number of polygons edge

        W = Image width

        H = Image height

        P = Number of image pixel with label

    """

    def __init__(self, labels: np.ndarray,
                 cords: np.ndarray,
                 prob: np.ndarray,
                 filename: str,
                 model: STARDIST_MODEL):
        """

        :param labels: Image with label. `Array[float, [H, W]]`
        :param cords: Coordinates. `Array[float, [N, 2, E]]`
        :param prob: Detected probability. `Array[float, N]`
        :param filename: filepath of image
        :param model: :attr:`STARDIST_MODEL`
        """
        self._labels = labels
        self._cords = cords
        self._prob = prob

        self._filename = filename
        self._model = model

    @classmethod
    def load(cls, file: PathLike) -> Self:
        """
        Load a stardist segmentation result

        :param file: stardist segmentation result ``.npz`` file
        :return: :class:`StarDistSegmentation`
        """
        dat = np.load(file, allow_pickle=True)
        return cls(labels=cls._reconstruct_labels_from_index_value(dat),
                   cords=dat['cords'],
                   prob=dat['prob'],
                   filename=dat['filename'],
                   model=dat['model'])

    @classmethod
    def _reconstruct_labels_from_index_value(cls, dat: NpzFile) -> np.ndarray:
        h, w = dat['shape']
        index = dat['index']
        value = dat['value']

        image = np.full((h, w), np.nan)
        for i, (hi, wi) in enumerate(index):
            image[hi, wi] = value[i]

        return image

    @property
    def n_segmentation(self) -> int:
        """number of segmented cells"""
        return len(self._prob)

    @property
    def width(self) -> int:
        """image width"""
        return self._labels.shape[1]

    @property
    def height(self) -> int:
        """image height"""
        return self._labels.shape[0]

    @property
    def filename(self) -> Path:
        """filepath of image"""
        return Path(self._filename)

    @property
    def labels(self) -> np.ndarray:
        """Image with label. `Array[float, [H, W]]`"""
        return self._labels

    @property
    def cords(self) -> np.ndarray:
        """Coordinates. `Array[float, [N, 2, E]]`"""
        return self._cords

    @property
    def prob(self) -> np.ndarray:
        """Detected probability. `Array[float, N]`"""
        return self._prob

    @property
    def points(self) -> np.ndarray:
        """Coordinates to points by simple XY average. `Array[float, [N, 2]]`"""
        return self.cords.mean(axis=2)

    @property
    def model(self) -> STARDIST_MODEL:
        """stardist model type"""
        return self._model

    def mask_probability(self, threshold: float):
        """masking probability for the results

        :param threshold: probability threshold
        """
        mx = self.prob >= threshold
        self._cords = self._cords[mx]
        self._prob = self._prob[mx]

    # noinspection PyTypeChecker
    def to_npz(self, output_file: PathLike) -> None:
        """
        Save ``filename``, ``cord``, ``prob``, ``point``, ``shape``, ``index``, ``index``, ``value`` as a npz file.

        shape: `Array[int, 2] in H,W`

        index: index with labels. `Array[int, [P, 2]]`

        value: label value in its index `Array[int, P]`

        :param output_file: output ``*.npz`` file path
        """
        shape = np.array(self.labels.shape)
        index, value = self._get_index_value()

        np.savez(output_file,
                 cords=self.cords,
                 prob=self.prob,
                 points=self.points,
                 shape=shape,
                 index=index,
                 value=value,
                 filename=str(self.filename),
                 model=self.model)

    def _get_index_value(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Get labeled pixel index and its value.

        :return: `Array[int, [P, 2]]` and `Array[float, P]`
        """
        labels = self.labels
        mask = ~np.isnan(labels)
        index = np.argwhere(mask)
        value = labels[mask]

        return np.array(index).astype(np.int_), np.array(value).astype(np.int_)

    # noinspection PyTypeChecker
    def to_roi(self, output_file: PathLike) -> None:
        """Covert segmented roi to point roi, and save it as ``.roi`` for imageJ.

        :param output_file: ``*.roi`` output file path"""
        from roifile import ImagejRoi, ROI_TYPE, ROI_OPTIONS

        points = np.fliplr(self.points)  # XY rotate in .roi format
        roi = ImagejRoi(
            roitype=ROI_TYPE.POINT,
            options=ROI_OPTIONS.PROMPT_BEFORE_DELETING | ROI_OPTIONS.SUB_PIXEL_RESOLUTION,
            n_coordinates=self.points.shape[0],
            integer_coordinates=points,
            subpixel_coordinates=points
        )

        roi.tofile(output_file)
