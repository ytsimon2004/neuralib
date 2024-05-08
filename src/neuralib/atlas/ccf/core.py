from __future__ import annotations

import abc
from pathlib import Path
from typing import NamedTuple, Final, Any, Literal

import attrs
import numpy as np
import polars as pl
from scipy.io import loadmat
from scipy.io.matlab import MatlabOpaque

from neuralib.atlas.util import PLANE_TYPE
from neuralib.atlas.view import SlicePlane, load_slice_view
from neuralib.util.util_type import PathLike
from neuralib.util.util_verbose import fprint
from neuralib.util.utils import uglob, joinn

__all__ = [
    'AbstractCCFDir',
    'CCFBaseDir',
    'CCFOverlapDir',
    #
    'CCFTransMatrix',
    'load_transform_matrix'
]

# ======================== #
# CCF DataFolder Structure #
# ======================== #

CCF_GLOB_TYPE = Literal[
    'zproj',
    'roi',
    'roi_cpose',
    'resize',
    'resize_overlap',
    'processed',
    'transformation_matrix',
    'transformation_img',
    'transformation_img_overlap'
]

CHANNEL_SUFFIX = Literal['r', 'g', 'b', 'merge', 'overlap']


class AbstractCCFDir(metaclass=abc.ABCMeta):
    """
    ::

        ANIMAL_001/ (root)
            ├── raw/ (optional)
            ├── zproj/
            │    └── ANIMAL_001_g*_s*_{channel}.tif
            ├── roi/
            │    └── ANIMAL_001_g*_s*_{channel}.roi
            ├── roi_cpose/
            │    └── ANIMAL_001_g*_s*_{channel}.roi
            ├── resize/ (src for the allenccf)
            │    ├── ANIMAL_001_g*_s*_resize.tif
            │    └── processed/
            │           ├── ANIMAL_001_g*_s*_resize_processed.tif
            │           └── transformations/
            │                 ├── ANIMAL_001_g*_s*_resize_processed_transformed.tif
            │                 ├── ANIMAL_001_g*_s*_resize_processed_transform_data.mat
            │                 └── labelled_regions/
            │                       ├── {*channel}_roitable.csv
            │                       └── parsed_data /
            │                             └── parsed_csv_merge.csv
            │
            └── output_files/ (for generate output fig)

    """
    def __new__(cls, root: PathLike,
                auto_mkdir: bool = True,
                with_overlap_sources: bool = True):

        if with_overlap_sources:
            return object.__new__(CCFOverlapDir)
        else:
            return object.__new__(CCFBaseDir)

    def __init__(self, root: PathLike,
                 auto_mkdir: bool = True,
                 with_overlap_sources: bool = True):
        """
        :param root:
        :param auto_mkdir:
        :param with_overlap_sources:
        """
        self.root: Final[Path] = root

        if auto_mkdir:
            self._init_folder_structure()

        self.with_overlap_sources: Final[bool] = with_overlap_sources

    @abc.abstractmethod
    def _init_folder_structure(self) -> None:
        pass

    @abc.abstractmethod
    def glob(self,
             glass_id: int,
             slice_id: int,
             glob_type: CCF_GLOB_TYPE,
             hemisphere: Literal['i', 'c'] | None = None,
             channel: CHANNEL_SUFFIX | None = None) -> Path:
        """

        :param glass_id: Glass slide number
        :param slice_id: Slice sequencing under a glass slide
            (i.e., zigzag from upper left -> bottom left ->-> bottom right)
        :param glob_type:
        :param hemisphere:
        :param channel:
        :return:
        """
        pass

    @abc.abstractmethod
    def get_transformation_matrix(self,
                                  glass_id: int,
                                  slice_id: int,
                                  plane_type: PLANE_TYPE) -> CCFTransMatrix:
        pass


class CCFBaseDir(AbstractCCFDir):
    """Base folder structure for 2dccf pipeline"""

    def _init_folder_structure(self):
        iter_dir = (
            self.raw_folder,
            self.resize_folder,
            self.roi_folder,
            self.cpose_roi_folder,
            self.zproj_folder,
            self.processed_folder,
            self.transformed_folder,
            self.labelled_roi_folder,
            self.parsed_data_folder,
            self.output_folder,
        )

        for d in iter_dir:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                fprint(f'auto make folder <{d.name}> for {self.animal}', vtype='io')

    @property
    def animal(self) -> str:
        return self.root.name

    @property
    def raw_folder(self) -> Path:
        return self.root / 'raw'

    @property
    def resize_folder(self) -> Path:
        return self.root / 'resize'

    @property
    def roi_folder(self) -> Path:
        return self.root / 'roi'

    @property
    def cpose_roi_folder(self) -> Path:
        return self.root / 'roi_cpose'

    @property
    def zproj_folder(self) -> Path:
        return self.root / 'zproj'

    # ========================================== #
    # CCF folder (MATLAB pipeline auto-generate) #
    # ========================================== #

    @property
    def processed_folder(self) -> Path:
        return self.resize_folder / 'processed'

    @property
    def transformed_folder(self) -> Path:
        return self.processed_folder / 'transformations'

    @property
    def labelled_roi_folder(self) -> Path:
        return self.transformed_folder / 'labelled_regions'

    @property
    def parsed_data_folder(self) -> Path:
        return self.labelled_roi_folder / 'parsed_data'

    @property
    def parse_csv(self) -> Path:
        return uglob(self.parsed_data_folder, 'parsed_csv_merge.csv')

    # ======= #
    # Outputs #
    # ======= #

    @property
    def output_folder(self) -> Path:
        return self.root / 'output_files'

    def figure_output(self, *suffix, sep='_') -> Path:
        ret = self.output_folder / joinn(sep, *suffix)
        return ret.with_suffix('.pdf')

    def csv_output(self, *suffix, sep='_') -> Path:
        ret = self.output_folder / joinn(sep, *suffix)
        return ret.with_suffix('.csv')

    # ========= #
    # File Glob #
    # ========= #

    def get_slice_id(self, glass_id: int,
                     slice_id: int,
                     hemisphere: Literal['i', 'c'] | None = None,
                     channel: CHANNEL_SUFFIX | None = None) -> str:

        ret = f'{self.animal}_{glass_id}_{slice_id}'
        if hemisphere is not None:
            ret += f'_{hemisphere}'
        if channel is not None:
            ret += f'_{channel}'
        return ret

    slice_name: str = None  # assign when glob

    def glob(self,
             glass_id: int,
             slice_id: int,
             glob_type: CCF_GLOB_TYPE, *,
             hemisphere: Literal['i', 'c'] | None = None,
             channel: CHANNEL_SUFFIX | None = None) -> Path | None:

        if glob_type in ('roi', 'roi_cpose', 'zproj'):
            self.slice_name = name = self.get_slice_id(glass_id, slice_id, hemisphere=hemisphere, channel=channel)
        else:
            self.slice_name = name = self.get_slice_id(glass_id, slice_id, hemisphere=hemisphere)

        #
        if glob_type == 'roi':
            return uglob(self.roi_folder, f'{name}*.roi')
        elif glob_type == 'roi_cpose':
            return uglob(self.cpose_roi_folder, f'{name}*cpose.roi')
        elif glob_type == 'zproj':
            return uglob(self.zproj_folder, f'{name}.*')
        elif glob_type == 'resize':
            return uglob(self.zproj_folder, f'{name}_resize.*')
        elif glob_type == 'transformation_matrix':
            return uglob(self.transformed_folder, f'{name}*.mat')
        elif glob_type == 'transformation_img':
            return uglob(self.transformed_folder, f'{name}*.tif')
        else:
            return

    def get_transformation_matrix(self,
                                  glass_id: int,
                                  slice_id: int,
                                  plane_type: PLANE_TYPE) -> CCFTransMatrix:
        return load_transform_matrix(
            self.glob(glass_id, slice_id, 'transformation_matrix'),
            plane_type
        )


class CCFOverlapDir(CCFBaseDir):
    """ 2dccf Folder structure for multiple sources overlap labeling
    For example: Dual tracing with different fluorescence protein, and tend to see the overlap channel counts"""

    def _init_folder_structure(self):
        super()._init_folder_structure()

        iter_overlap = (
            self.resize_overlap_folder,
            self.processed_folder,
            self.transformed_folder_overlap
        )
        for d in iter_overlap:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                fprint(f'auto make folder <{d.name}> for {self.animal}', vtype='io')

    @property
    def resize_overlap_folder(self) -> Path:
        """for double lapping channel
        since maximal 2 channels for roi detection,
        then need extra folder use pseudo-color
        """
        return self.root / 'resize_overlap'

    @property
    def processed_folder_overlap(self) -> Path:
        return self.resize_overlap_folder / 'processed'

    @property
    def transformed_folder_overlap(self) -> Path:
        return self.processed_folder_overlap / 'transformations'

    def glob(self,
             glass_id: int,
             slice_id: int,
             glob_type: CCF_GLOB_TYPE, *,
             hemisphere: Literal['i', 'c'] | None = None,
             channel: CHANNEL_SUFFIX | None = None) -> Path | None:

        ret = super().glob(glass_id, slice_id, glob_type, hemisphere=hemisphere, channel=channel)

        if ret is None:
            if glob_type == 'resize_overlap':
                return uglob(self.resize_overlap_folder, f'{self.slice_name}_resize_overlap.*')
            elif glob_type == 'transformation_img_overlap':
                return uglob(self.transformed_folder_overlap, f'{self.slice_name}*.tif')

        return


class CCFSagittalDir(AbstractCCFDir):
    # TODO
    pass


# ===================== #
# Transformation matrix #
# ===================== #

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
    def angle_xy(self) -> tuple[float, float]:
        """angle shifting for xy axis (width, height) for a specific plane type"""
        angle = self.allen_location[1]
        return angle[0], angle[1]


def load_transform_matrix(filepath: PathLike,
                          plane_type: PLANE_TYPE,
                          resolution: int = 10,
                          default_name: str = 'test') -> CCFTransMatrix:
    """
    matrix for image transformation

    :param filepath:
    :param plane_type
    :param resolution
    :param default_name
    :return:
    """
    mat = loadmat(filepath, squeeze_me=True, struct_as_record=False)['save_transform']
    f = Path(filepath).name

    try:
        name = f[:f.index('_resize')]
    except ValueError:
        name = default_name

    return CCFTransMatrix(name,
                          MatMatrix(
                              mat.allen_location,
                              mat.transform,
                              mat.transform_points
                          ),
                          plane_type,
                          resolution=resolution)


@attrs.frozen
class CCFTransMatrix:
    slice_id: str
    matrix: MatMatrix
    plane_type: PLANE_TYPE
    resolution: int = attrs.field(kw_only=True, default=10)

    @property
    def slice_index(self) -> int:
        return self.matrix.slice_index

    @property
    def angle_xy(self) -> tuple[float, float]:
        return self.matrix.angle_xy

    def get_slice_plane(self) -> SlicePlane:
        ret = (load_slice_view('ccf_template', self.plane_type, allen_annotation_res=self.resolution)
               .plane_at(self.slice_index))

        if np.any([a != 0 for a in self.matrix.angle_xy]):
            ret = ret.with_angle_offset(-self.matrix.angle_xy[0], self.matrix.angle_xy[1])

        return ret

    def matrix_info(self, to_polars: bool = True) -> dict[str, Any] | pl.DataFrame:
        ret = {
            'slice_id': self.slice_id,
            'plane_type': self.plane_type,
            'atlas_resolution': self.resolution,
            'slice_index': self.slice_index,
            'reference_value': self.get_slice_plane().reference_value,
            'dw': self.angle_xy[0],
            'dh': self.angle_xy[1]
        }

        if to_polars:
            ret = pl.DataFrame(ret)

        return ret
