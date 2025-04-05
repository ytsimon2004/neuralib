from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Any

import attrs
import numpy as np
import polars as pl
from scipy.io import loadmat
from scipy.io.matlab import MatlabOpaque

from neuralib.atlas.typing import PLANE_TYPE
from neuralib.atlas.view import SlicePlane, get_slice_view
from neuralib.typing import PathLike

__all__ = [
    'CCFTransMatrix',
    'load_transform_matrix'
]


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
        return self.matrix.delta_values[1], self.matrix.delta_values[0]

    def get_slice_plane(self) -> SlicePlane:
        """get slice plane"""
        ret = (
            get_slice_view('reference', self.plane_type, resolution=self.resolution)
            .plane_at(self.slice_index)
        )

        dw = self.delta_xy[0]
        dh = self.delta_xy[1]

        ret = ret.with_offset(
            dw=dw + 1 if dw != 0 else 0,  # avoid index err
            dh=dh + 1 if dh != 0 else 0,  # avoid index err
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
