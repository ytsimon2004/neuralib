from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Any, get_args

import attrs
import cv2
import imageio.v3 as iio
import numpy as np
import polars as pl
from scipy.io import loadmat
from scipy.io.matlab import MatlabOpaque

from neuralib.atlas.typing import PLANE_TYPE
from neuralib.atlas.view import SlicePlane, get_slice_view
from neuralib.imglib.transform import apply_transformation
from neuralib.typing import PathLike

__all__ = [
    'SLICE_DIMENSION_10um',
    'CCFTransMatrix',
    'load_transform_matrix',
    'slice_transform_helper'
]

SLICE_DIMENSION_10um: dict[PLANE_TYPE, tuple[int, int]] = {
    'coronal': (1140, 800),
    'sagittal': (1320, 800)
}


def load_transform_matrix(filepath: PathLike,
                          plane_type: PLANE_TYPE,
                          resolution: int = 10,
                          default_name: str = 'test') -> CCFTransMatrix:
    """
    matrix for image transformation

    :param filepath: transformation matrix .mat file
    :param plane_type: ``PLANE_TYPE`` {'coronal', 'sagittal', 'transverse'}
    :param resolution: atlas resolution
    :param default_name: name of the ``CCFTransMatrix``
    :return: ``CCFTransMatrix``
    """
    mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)['save_transform']
    f = Path(filepath).name

    try:
        name = f[:f.index('_resize')]
    except ValueError:
        name = default_name

    return CCFTransMatrix(
        name,
        MatMatrix(mat.allen_location, mat.transform, mat.transform_points),
        plane_type,
        resolution=resolution
    )


class MatMatrix(NamedTuple):
    allen_location: np.ndarray
    """abnormal parsing in loadmat. [996 array([  0, -12], dtype=int16)]"""
    transform: MatlabOpaque
    """mat meta info. currently not used"""
    transform_points: np.ndarray
    """[(P, 2), (P, 2)] transformation point for 2dccf GUI site view"""

    @property
    def slice_index(self) -> int:
        return int(self.allen_location[0])

    # noinspection PyTypeChecker
    @property
    def delta_values(self) -> tuple[int, int]:
        """shifting values for yx axis (height, width) for a specific plane type.

        Coronal slice: [0]:DV, [1]:ML

        Sagittal Slice: [0]:DV, [1]:AP

        """
        delta = self.allen_location[1]
        return delta[0], delta[1]


@attrs.frozen
class CCFTransMatrix:
    """matrix for image transformation"""

    slice_id: str
    """slice name id"""
    matrix: MatMatrix
    """MatMatrix"""
    plane_type: PLANE_TYPE
    """PLANE_TYPE"""
    resolution: int = attrs.field(kw_only=True, default=10)
    """resolution in um"""

    @property
    def slice_index(self) -> int:
        """slice index"""
        return self.matrix.slice_index

    @property
    def delta_xy(self) -> tuple[int, int]:
        """map delta value to xy slice view"""
        return int(self.matrix.delta_values[1]), int(self.matrix.delta_values[0])

    def get_slice_plane(self) -> SlicePlane:
        """get slice plane"""
        dw = self.delta_xy[0]
        dh = self.delta_xy[1]

        ret = (
            get_slice_view('reference', self.plane_type, resolution=self.resolution)
            .plane_at(self.slice_index)
            .with_offset(
                dw=dw + 1 if dw != 0 else 0,  # avoid index err
                dh=dh + 1 if dh != 0 else 0,  # avoid index err
            )
        )

        return ret

    def matrix_info(self, to_polars: bool = True) -> dict[str, Any] | pl.DataFrame:
        """
        transform matrix information dict

        :param to_polars: to polars dataframe
        :return:
        """
        ret = {
            'slice_id': self.slice_id,
            'plane_type': self.plane_type,
            'atlas_resolution': self.resolution,
            'slice_index': self.slice_index,
            'reference_value': self.get_slice_plane().reference_value,
            'dw': self.delta_xy[0],
            'dh': self.delta_xy[1]
        }

        if to_polars:
            ret = pl.DataFrame(ret)

        return ret


def slice_transform_helper(raw_image: PathLike | np.ndarray,
                           trans_matrix: np.ndarray | PathLike, *,
                           plane_type: PLANE_TYPE = 'coronal',
                           flip_lr: bool = False,
                           flip_ud: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Transforms an input image according to the specified transformation matrix,
    plane orientation, and optional flipping parameters. This function reads a raw
    image, optionally flips it horizontally or vertically, applies a transformation
    matrix, and resizes it to a defined dimension based on the plane type. The
    result is a tuple containing the raw image and the transformed image.

    :param raw_image: Path to the raw input image or image array.
    :param trans_matrix: Transformation matrix to apply to the image. It can be a
        numpy array or the path to a valid file containing the matrix. Supported
        files are `.mat` for MATLAB files and `.npy` for NumPy files.
    :param plane_type: Defines the anatomical plane for transformation.
        Defaults to 'coronal'.
    :param flip_lr: Boolean flag to determine whether to flip the image
        horizontally (left to right).
    :param flip_ud: Boolean flag to determine whether to flip the image
        vertically (up to down).
    :return: A tuple containing two numpy arrays: the raw image as processed,
        and the transformed image.
    """
    if isinstance(raw_image, np.ndarray):
        pass
    elif isinstance(raw_image, get_args(PathLike)):
        raw_image = iio.imread(raw_image)
    else:
        raise TypeError(f"Unsupported type {type(raw_image)}")

    #
    if flip_ud:
        raw_image = np.flipud(raw_image)

    if flip_lr:
        raw_image = np.fliplr(raw_image)

    #
    if isinstance(trans_matrix, get_args(PathLike)):
        s = Path(trans_matrix).suffix
        if s == '.mat':
            mtx = loadmat(trans_matrix)['t'].T
        elif s == '.npy':
            mtx = np.load(trans_matrix)
        else:
            raise ValueError(f'invalid file type for the transformation matrix: {s}')
    else:
        mtx = trans_matrix

    resize = cv2.resize(raw_image, SLICE_DIMENSION_10um[plane_type])
    transform_image = apply_transformation(resize, mtx)

    return raw_image, transform_image
