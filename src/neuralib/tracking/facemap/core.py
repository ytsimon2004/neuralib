from __future__ import annotations

import pickle
from typing import TypedDict

import h5py
import numpy as np
import polars as pl
from typing_extensions import Self

from neuralib.typing import PathLike
from neuralib.util.dataframe import DataFrameWrapper
from neuralib.util.utils import uglob

__all__ = [
    'read_facemap',
    'KeyPoint',
    'FaceMapResult',
    'KeyPointDataFrame',
    'SVDVariables',
    'KeyPointsMeta',
    'PupilDict',
    'RoiDict'
]


def read_facemap(directory: PathLike) -> FaceMapResult:
    """loading facemap result

    :param directory: facemap output directory
    """
    return FaceMapResult.from_directory(directory)


KeyPoint = str
"""keypoint name"""


class FaceMapResult:
    """Facemap result container"""

    def __init__(self, svd: SVDVariables | None,
                 meta: KeyPointsMeta | None,
                 data: h5py.Group | None,
                 with_keypoints: bool):
        """
        :param svd: attr:`~neuralib.tracking.facemap.core.SVDVariables`
        :param meta: attr:`~neuralib.tracking.facemap.core.KeyPointsMeta`
        :param data: facemap data result
        :param with_keypoints: whether it has keypoint tracking
        """
        self.svd = svd
        self.meta = meta
        self.data = data

        self._with_keypoints = with_keypoints

    @classmethod
    def from_directory(cls, directory: PathLike) -> Self:
        """
        init class loading from a directory

        :param directory: Facemap output directory
        """
        # svd
        try:
            svd_path = uglob(directory, '*.npy')
        except FileNotFoundError:
            svd = None
        else:
            svd = np.load(svd_path, allow_pickle=True).item()

        # meta
        try:
            meta_path = uglob(directory, '*.pkl')
        except FileNotFoundError:
            meta = None
            data = None
            keypoints = False
        else:
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)

            data_path = uglob(directory, '*.h5')
            data = h5py.File(data_path)['Facemap']
            keypoints = True

        return cls(svd, meta, data, keypoints)

    @property
    def with_keypoint(self) -> bool:
        return self._with_keypoints

    # ============== #
    # Pupil Tracking #
    # ============== #

    def get_pupil(self) -> PupilDict:
        """pupil tracking result

        :raises RuntimeError: If no pupil data is available.
        """
        try:
            pupil: list[PupilDict] = self.svd['pupil']
        except KeyError:
            raise RuntimeError('no pupil data found')
        else:
            return pupil[0]

    def get_pupil_area(self) -> np.ndarray:
        """pupil area. `Array[float, F]`"""
        return self.get_pupil()['area_smooth']

    def get_pupil_center_of_mass(self) -> np.ndarray:
        """center of mass of pupil tracking. `Array[float, [F, 2]]`"""
        return self.get_pupil()['com_smooth']

    def get_pupil_location_movement(self) -> np.ndarray:
        """Calculate the Euclidean distance from the origin for each point in a 2D array. `Array[float, F]`"""
        com = self.get_pupil_center_of_mass()
        return np.sqrt(np.sum(com ** 2, axis=1))

    def get_blink(self) -> np.ndarray:
        """eye blinking array. `Array[float, F]`

        :raises RuntimeError: If no blink data is available.
        """
        try:
            ret = self.svd['blink']
        except KeyError:
            raise RuntimeError('no blink data found')
        else:
            return ret[0]

    # ========= #
    # Keypoints #
    # ========= #

    @property
    def keypoints(self) -> list[KeyPoint]:
        """list of all keypoint name"""
        return list(self.data.keys())

    def get(self, *keypoint) -> KeyPointDataFrame:
        """get keypoint(s) dataframe"""
        if len(keypoint) == 1:
            return self._get(keypoint[0])
        else:
            ret = [self._get(k).dataframe() for k in keypoint]
            return KeyPointDataFrame(pl.concat(ret))

    def _get(self, keypoint: KeyPoint) -> KeyPointDataFrame:
        x = np.array(self.data[keypoint]['x'])
        y = np.array(self.data[keypoint]['y'])
        llh = np.array(self.data[keypoint]['likelihood'])
        df = pl.DataFrame({'x': x, 'y': y, 'likelihood': llh}).with_columns(pl.lit(keypoint).alias('keypoint'))
        return KeyPointDataFrame(df)


class KeyPointDataFrame(DataFrameWrapper):
    """
    Dataframe with ``x``, ``y``, ``likelihood`` and ``keypoint`` columns ::

        ┌────────────┬────────────┬────────────┬───────────┐
        │ x          ┆ y          ┆ likelihood ┆ keypoint  │
        │ ---        ┆ ---        ┆ ---        ┆ ---       │
        │ f32        ┆ f32        ┆ f32        ┆ str       │
        ╞════════════╪════════════╪════════════╪═══════════╡
        │ 374.102081 ┆ 199.159668 ┆ 0.777443   ┆ eye(back) │
        │ 373.785919 ┆ 199.425873 ┆ 0.787424   ┆ eye(back) │
        │ 374.075867 ┆ 199.507111 ┆ 0.779713   ┆ eye(back) │
        │ 374.028473 ┆ 199.359955 ┆ 0.761724   ┆ eye(back) │
        │ 374.222382 ┆ 199.777466 ┆ 0.770329   ┆ eye(back) │
        │ …          ┆ …          ┆ …          ┆ …         │
        │ 317.318756 ┆ 285.396912 ┆ 0.596486   ┆ mouth     │
        │ 318.163696 ┆ 285.492676 ┆ 0.589684   ┆ mouth     │
        │ 317.758606 ┆ 285.560425 ┆ 0.603126   ┆ mouth     │
        │ 317.453491 ┆ 285.572235 ┆ 0.573179   ┆ mouth     │
        │ 317.976196 ┆ 285.477051 ┆ 0.58359    ┆ mouth     │
        └────────────┴────────────┴────────────┴───────────┘

    """

    def __init__(self, df: pl.DataFrame):
        self._df = df

    def __repr__(self):
        return repr(self.dataframe())

    def dataframe(self, dataframe: pl.DataFrame = None, may_inplace=True):
        if dataframe is None:
            return self._df
        else:
            return KeyPointDataFrame(dataframe)

    def to_zscore(self) -> Self:
        """
        xy to zscore

        :return:
        """
        return self.with_columns([
            ((pl.col('x') - pl.col('x').mean()) / pl.col('x').std()).alias('x'),
            ((pl.col('y') - pl.col('y').mean()) / pl.col('y').std()).alias('y'),
        ])

    def with_outlier_filter(self, filter_window: int = 15,
                            baseline_window: int = 50,
                            max_spike: int = 25,
                            max_diff: int = 25) -> Self:
        """
        with outlier filter

        :param filter_window:
        :param baseline_window:
        :param max_spike:
        :param max_diff:
        :return:
        """
        from .util import filter_outliers
        x, y = filter_outliers(np.array(self['x']), np.array(self['y']), filter_window, baseline_window, max_spike, max_diff)

        return self.dataframe(pl.DataFrame({
            'x': x,
            'y': y,
            'likelihood': self['likelihood'],
            'keypoint': self['keypoint']
        }))


class PupilDict(TypedDict):
    """
    Pupil data dict

    `Dimension parameters`:

        F: number pf frames
    """
    area: np.ndarray
    """`Array[float, F]`"""
    com: np.ndarray
    """center of maze in XY. `Array[float, [F, 2]]`"""
    axdir: np.ndarray
    """`Array[float, [F, 2, 2]]`"""
    axlen: np.ndarray
    """`Array[float, [F, 2]]`"""
    area_smooth: np.ndarray
    """`Array[float, F]`"""
    com_smooth: np.ndarray
    """`Array[float, [F, 2]]`"""


class RoiDict(TypedDict, total=False):
    """Roi Dict"""
    rind: int
    rtype: str
    iROI: int
    ivid: int
    color: tuple[float, float, float]
    yrange: np.ndarray
    xrange: np.ndarray
    saturation: float
    pupil_sigma: float
    ellipse: np.ndarray
    yrange_bin: np.ndarray
    xrange_bin: np.ndarray


class SVDVariables(TypedDict, total=False):
    """SVD output from facemap
    .. seealso:: `<http://facemap.readthedocs.io/en/stable/outputs.html#roi-and-svd-processing>`_"""
    filenames: list[str]
    save_path: str
    Ly: list[int]
    Lx: list[int]
    sbin: int
    fullSVD: bool
    save_mat: bool
    Lybin: np.ndarray
    Lxbin: np.ndarray
    sybin: np.ndarray
    sxbin: np.ndarray
    LYbin: int
    LXbin: int
    avgframe: list[np.ndarray]
    avgmotion: list[np.ndarray]
    avgframe_reshape: np.ndarray
    avgmotion_reshape: np.ndarray
    motion: list[np.ndarray]
    motSv: list[np.ndarray]
    movSv: list[np.ndarray]
    motMask: list[int]
    movMask: list[int]
    motMask_reshape: list[int]
    movMask_reshape: list[int]
    motSVD: list[np.ndarray]
    movSVD: list[np.ndarray]
    pupil: list[PupilDict]
    running: list[np.ndarray]
    blink: list[np.ndarray]
    rois: list[RoiDict]
    sy: np.ndarray
    sx: np.ndarray


class KeyPointsMeta(TypedDict):
    """ Keypoint meta
    .. seealso:: `<https://facemap.readthedocs.io/en/stable/outputs.html#keypoints-processing>`_"""
    batch_size: int
    image_size: tuple[list[int], ...]
    bbox: tuple[int, ...]
    total_frames: int
    bodyparts: list[str]
    inference_speed: float
