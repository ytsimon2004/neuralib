from __future__ import annotations

from pathlib import Path

import numpy as np
from typing_extensions import Self

from neuralib.typing import PathLike

__all__ = [
    'read_cellpose',
    'cellpose_point_roi_helper',
    'CellposeSegmentation'
]


def read_cellpose(file: PathLike) -> CellposeSegmentation:
    """
    Read a cellpose segmentation result file

    :param file: cellpose segmentation result ``.npy`` file
    :return: :class:`CellposeSegmentation`
    """
    return CellposeSegmentation.load(file)


def cellpose_point_roi_helper(file: PathLike, output: PathLike) -> None:
    """
    Read a cellpose segmentation result and convert the segmentation result to point coordinates

    :param file: cellpose segmentation result ``.npy`` file
    :param output: ``*.roi`` output file path
    """
    CellposeSegmentation.load(file).to_roi(output)


class CellposeSegmentation:
    """`Cellpose <https://github.com/MouseLand/cellpose>`_ segmentation results

    `Dimension parameters`:

        N = Number of segmented cells

        W = Image width

        H = Image height

    .. seealso::

        `Cellpose Native Doc <https://cellpose.readthedocs.io/en/latest/outputs.html#seg-npy-output>`_

    """

    def __init__(self, outlines, masks, chan_choose, ismanual, flows, diameter, filename):
        self._outlines = outlines
        self._masks = masks
        self._chan_choose = chan_choose
        self._is_manual = ismanual
        self._flows = flows
        self._diameter = diameter

        self._filename = filename

    @classmethod
    def load(cls, file: PathLike) -> Self:
        """
        Load a cellpose segmentation result

        :param file: cellpose segmentation result ``.npy`` file
        :return: :class:`CellposeSegmentation`
        """
        dat = np.load(file, allow_pickle=True).item()
        return cls(**dat)

    @property
    def n_segmentation(self) -> int:
        """number of segmented cells"""
        return len(self._is_manual)

    @property
    def width(self):
        """image width"""
        return self._outlines.shape[1]

    @property
    def height(self):
        """image height"""
        return self._outlines.shape[0]

    @property
    def filename(self) -> Path:
        """filepath of image"""
        return Path(self._filename)

    @property
    def outlines(self) -> np.ndarray:
        """outlines of ROIs (0 = NO outline; 1,2,… = outline labels). `Array[uint16, [H, W]]`"""
        return self._outlines

    @property
    def masks(self) -> np.ndarray:
        """each pixel in the image is assigned to an ROI (0 = NO ROI; 1,2,… = ROI labels). `Array[uint16, [H, W]]` """
        return self._masks

    @property
    def chan_choose(self) -> list[int]:
        """channels that you chose in GUI (0=gray/none, 1=red, 2=green, 3=blue)"""
        return self._chan_choose

    @property
    def flows(self) -> list[np.ndarray]:
        """
        flows[0] is XY flow in RGB

        flows[1] is the cell probability in range 0-255 instead of -10.0 to 10.0

        flows[2] is Z flow in range 0-255 (if it exists, otherwise zeros),

        flows[3] is [dY, dX, cellprob] (or [dZ, dY, dX, cellprob] for 3D), flows[4] is pixel destinations (for internal use)
        """
        return self._flows

    @property
    def diameter(self) -> float:
        """cell body diameter"""
        return self._diameter

    @property
    def is_manual(self) -> np.ndarray:
        """whether or not mask k was manually drawn or computed by the cellpose algorithm. `Array[bool, N]`"""
        return self._is_manual

    @property
    def nan_masks(self) -> np.ndarray:
        """value 0 in :attr:`CellposeSegmentation.masks` to nan"""
        masks = self.masks.copy().astype(np.float_)
        masks[masks == 0] = np.nan

        return masks

    @property
    def nan_outlines(self) -> np.ndarray:
        """value 0 in :attr:`CellposeSegmentation.outlines` to nan"""
        outlines = self.outlines.copy().astype(np.float_)
        outlines[outlines == 0] = np.nan

        return outlines

    @property
    def points(self) -> np.ndarray:
        """Calculate center of each segmented area in XY pixel. `Array[int, [N, 2]]`"""
        centers = self._calculate_centers()
        return np.round(centers).astype(int)

    # noinspection PyTypeChecker
    def to_roi(self, output_file: PathLike):
        """
        Covert segmented roi to point roi, and save it as ``.roi`` for imageJ.

        :param output_file: ``*.roi`` output file path
        """
        from roifile import ImagejRoi, ROI_TYPE, ROI_OPTIONS

        if Path(output_file).suffix != '.roi':
            raise ValueError('output file must have .roi extension')

        points = np.fliplr(self.points)  # XY rotate in .roi format
        roi = ImagejRoi(
            roitype=ROI_TYPE.POINT,
            options=ROI_OPTIONS.PROMPT_BEFORE_DELETING | ROI_OPTIONS.SUB_PIXEL_RESOLUTION,
            n_coordinates=self.points.shape[0],
            integer_coordinates=points,
            subpixel_coordinates=points
        )

        roi.tofile(output_file)

    def _calculate_centers(self):
        """calculate center of each segmented area in XY pixel"""
        labels = np.unique(self.masks)
        labels = labels[labels != 0]  # remove background

        n_neurons = len(labels)
        centers = np.zeros((n_neurons, 2))
        for i, label in enumerate(labels):
            segment_coords = np.argwhere(self.masks == label)
            center = segment_coords.mean(axis=0)
            centers[i] = center

        return centers
