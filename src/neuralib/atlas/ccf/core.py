import abc
from pathlib import Path
from typing import NamedTuple, Final, Any, Literal, get_args, Iterable

import attrs
import numpy as np
import polars as pl
from neuralib.atlas.typing import HEMISPHERE_TYPE
from neuralib.atlas.util import PLANE_TYPE
from neuralib.atlas.view import SlicePlane, load_slice_view
from neuralib.typing import PathLike
from neuralib.util.utils import uglob, joinn
from neuralib.util.verbose import fprint
from scipy.io import loadmat
from scipy.io.matlab import MatlabOpaque

__all__ = [
    #
    'CCF_GLOB_TYPE',
    'CHANNEL_SUFFIX',
    #
    'AbstractCCFDir',
    'CoronalCCFDir',
    'CoronalCCFOverlapDir',
    'SagittalCCFDir',
    'SagittalCCFOverlapDir',
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
            ├── resize/ (src for the allenccf, if sagittal slice, could be resize_contra and resize_ipsi)
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
            ├── resize_*_overlap/ (optional, same structure as **resize**, for dual channels labeling)
            │
            └── output_files/ (for generate output fig)

    """

    def __new__(
            cls,
            root: PathLike,
            with_overlap_sources: bool = True,
            plane_type: PLANE_TYPE = 'coronal',
            hemisphere_type: HEMISPHERE_TYPE | None = None,
            auto_mkdir: bool = True,
    ):

        if plane_type == 'coronal':
            if with_overlap_sources:
                return object.__new__(CoronalCCFOverlapDir)
            else:
                return object.__new__(CoronalCCFDir)

        elif plane_type == 'sagittal':

            if hemisphere_type is None or hemisphere_type not in get_args(HEMISPHERE_TYPE):
                raise ValueError(f'invalid hemisphere_type for sagittal dir: {hemisphere_type}')

            #
            if with_overlap_sources:
                return object.__new__(SagittalCCFOverlapDir)
            else:
                return object.__new__(SagittalCCFDir)

        else:
            raise ValueError(f'invalid plane type: {plane_type}')

    def __init__(
            self,
            root: PathLike,
            with_overlap_sources: bool = True,
            plane_type: PLANE_TYPE = 'coronal',
            hemisphere_type: HEMISPHERE_TYPE | None = None,
            auto_mkdir: bool = True,
    ):
        r"""

        :param root: Root path (i.e., \*/ANIMAL_001)
        :param with_overlap_sources: If there is overlap channel labeling (for dir making).
        :param plane_type: {'coronal', 'sagittal', 'transverse'}
        :param hemisphere_type: {'ipsi', 'contra', 'both'}
        :param auto_mkdir: If auto make folder structure for the pipeline.
        """
        self.root: Final[Path] = root
        self.with_overlap_sources = with_overlap_sources
        self.plane_type: Final[PLANE_TYPE] = plane_type
        self.hemisphere: HEMISPHERE_TYPE | None = hemisphere_type if plane_type == 'sagittal' else None

        if auto_mkdir:
            self._init_folder_structure()

    def __len__(self):
        """number of slices"""
        return len(list(self.resize_folder.glob('*.tif')))

    def __iter__(self):
        return self.resize_folder.glob('*.tif')

    @abc.abstractmethod
    def _init_folder_structure(self) -> None:
        pass

    slice_name: str = None  # assign when glob

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

    @property
    @abc.abstractmethod
    def resize_folder(self) -> Path:
        pass

    # ============================ #
    # Default Dir for the pipeline #
    # ============================ #

    @property
    def default_iter_dir(self) -> Iterable[Path]:
        return [
            self.raw_folder,
            self.resize_folder,
            self.roi_folder,
            self.cpose_roi_folder,
            self.zproj_folder,
            self.processed_folder,
            self.transformed_folder,
            self.labelled_roi_folder,
            self.parsed_data_folder,
            self.output_folder
        ]

    @property
    def animal(self) -> str:
        return self.root.name

    @property
    def raw_folder(self) -> Path:
        return self.root / 'raw'

    @property
    def roi_folder(self) -> Path:
        return self.root / 'roi'

    @property
    def cpose_roi_folder(self) -> Path:
        return self.root / 'roi_cpose'

    @property
    def zproj_folder(self) -> Path:
        return self.root / 'zproj'

    # =============================== #
    # Dual Channel Overlap (Optional) #
    # =============================== #

    @property
    def resize_overlap_folder(self) -> Path | None:
        if not self.with_overlap_sources:
            raise ValueError('')
        return

    @property
    def processed_folder_overlap(self) -> Path | None:
        if not self.with_overlap_sources:
            raise ValueError('')
        return

    @property
    def transformed_folder_overlap(self) -> Path | None:
        if not self.with_overlap_sources:
            raise ValueError('')
        return

    @property
    def labelled_roi_folder_overlap(self) -> Path | None:
        if not self.with_overlap_sources:
            raise ValueError('')
        return

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
        return self.parsed_data_folder / 'parsed_csv_merge.csv'

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

    @property
    def roi_atlas_output(self) -> Path:
        return self.output_folder / 'roiatlas'

    @property
    def roi_atlas_ibl_output(self) -> Path:
        return self.output_folder / 'roiatlas_ibl'

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

    def get_transformation_matrix(self, glass_id: int,
                                  slice_id: int,
                                  plane_type: PLANE_TYPE) -> 'CCFTransMatrix':
        return load_transform_matrix(
            self.glob(glass_id, slice_id, 'transformation_matrix'),
            plane_type
        )


# ============= #
# Coronal Slice #
# ============= #

class CoronalCCFDir(AbstractCCFDir):
    """Base folder structure for 2dccf pipeline"""

    def _init_folder_structure(self):
        for d in self.default_iter_dir:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                fprint(f'auto make folder <{d.name}> for {self.animal}', vtype='io')

    @property
    def resize_folder(self) -> Path:
        return self.root / 'resize'

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
            return uglob(self.resize_folder, f'{name}_resize.*')
        elif glob_type == 'transformation_matrix':
            return uglob(self.transformed_folder, f'{name}*.mat')
        elif glob_type == 'transformation_img':
            return uglob(self.transformed_folder, f'{name}*.tif')
        else:
            return


class CoronalCCFOverlapDir(CoronalCCFDir):
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
        """for double labeling channel
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

    @property
    def labelled_roi_folder_overlap(self) -> Path:
        return self.transformed_folder_overlap / 'labelled_regions'

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

        return ret


# ============== #
# Sagittal Slice #
# ============== #

class SagittalCCFDir(AbstractCCFDir):

    def _init_folder_structure(self) -> None:
        for d in self.default_iter_dir:
            if not d.exists():
                d.mkdir(parents=True, exist_ok=True)
                fprint(f'auto make folder <{d.name}> for {self.animal}', vtype='io')

    def glob(self, glass_id: int, slice_id: int, glob_type: CCF_GLOB_TYPE, hemisphere: Literal['i', 'c'] | None = None,
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

    @property
    def resize_folder(self) -> Path:
        if self.hemisphere == 'ipsi':
            return self.root / 'resize_ipsi'
        elif self.hemisphere == 'contra':
            return self.root / 'resize_contra'
        elif self.hemisphere == 'both':
            return self.root / 'resize'
        else:
            raise ValueError('')


class SagittalCCFOverlapDir(SagittalCCFDir):
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
        """for double labeling channel
        since maximal 2 channels for roi detection,
        then need extra folder use pseudo-color
        """
        if self.hemisphere == 'ipsi':
            return self.root / 'resize_ipsi_overlap'
        elif self.hemisphere == 'contra':
            return self.root / 'resize_contra_overlap'
        elif self.hemisphere == 'both':
            return self.root / 'resize_overlap'
        else:
            raise ValueError('')

    @property
    def processed_folder_overlap(self) -> Path:
        return self.resize_overlap_folder / 'processed'

    @property
    def transformed_folder_overlap(self) -> Path:
        return self.processed_folder_overlap / 'transformations'

    @property
    def labelled_roi_folder_overlap(self) -> Path:
        return self.transformed_folder_overlap / 'labelled_regions'

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
    def delta_values(self) -> tuple[int, int]:
        """shifting values for yx axis (height, width) for a specific plane type.

        Coronal slice: [0]:DV, [1]:ML

        Sagittal Slice: [0]:DV, [1]:AP

        """
        delta = self.allen_location[1]
        return delta[0], delta[1]


def load_transform_matrix(filepath: PathLike,
                          plane_type: PLANE_TYPE,
                          resolution: int = 10,
                          default_name: str = 'test') -> 'CCFTransMatrix':
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
    def delta_xy(self) -> tuple[int, int]:
        """map delta value to xy slice view"""
        return self.matrix.delta_values[1], self.matrix.delta_values[0]

    def get_slice_plane(self) -> SlicePlane:
        ret = (
            load_slice_view('ccf_template', self.plane_type, allen_annotation_res=self.resolution)
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
