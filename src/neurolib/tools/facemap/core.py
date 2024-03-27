from __future__ import annotations

import pickle
from typing import TypedDict, Generator, overload, final

import attrs
import h5py
import numpy as np
from typing_extensions import TypeAlias, Self, Literal

from neurolib.stimpy.event import CamEvent
from neurolib.stimpy.stimpy_core import RiglogData
from neurolib.util.cli_args import CliArgs

__all__ = [
    'TRACK_TYPE',
    'KeyPoint',
    'FaceMapResult',
    'KeyPointTrack',
]

from neurolib.util.util_type import PathLike
from neurolib.util.util_verbose import fprint
from neurolib.util.utils import uglob

TRACK_TYPE = Literal['keypoints', 'pupil']


class PupilDict(TypedDict):
    area: np.ndarray
    """(N,)"""
    com: np.ndarray
    """center of maze (N, 2)"""
    axdir: np.ndarray
    """(N, 2, 2)"""
    axlen: np.ndarray
    """(N, 2)"""
    area_smooth: np.ndarray
    """(N,)"""
    com_smooth: np.ndarray
    """(N, 2)"""


class RoiDict(TypedDict, total=False):
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
    """http://facemap.readthedocs.io/en/stable/outputs.html#roi-and-svd-processing"""
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
    running: list[np.ndarray]  # TODO check
    blink: list[np.ndarray]
    rois: list[RoiDict]  # TODO check
    sy: np.ndarray
    sx: np.ndarray


class KeyPointsMeta(TypedDict):
    """https://facemap.readthedocs.io/en/stable/outputs.html#keypoints-processing"""
    batch_size: int
    image_size: tuple[list[int], ...]
    bbox: tuple[int, ...]
    total_frames: int
    bodyparts: list[str]
    inference_speed: float


# ============== #
# FaceMap Result #
# ============== #

KeyPoint: TypeAlias = str


@final
class FaceMapResult:
    __slots__ = ('svd', 'meta', 'data', 'rig',
                 '_track_type', '_with_keypoints')

    def __init__(
            self,
            rig: RiglogData,
            svd: SVDVariables,
            meta: KeyPointsMeta | None,
            data: h5py.Group | None,
            track_type: 'TRACK_TYPE',
            with_keypoints: bool,
    ):
        """

        :param rig: Stimpy .riglog (for time sync)
        :param svd: SVD processing outputs
        :param meta: Optional for Keypoints processing (result)
        :param data: Optional for Keypoints processing (config)
        """
        self.rig = rig
        self.svd = svd
        self.meta = meta
        self.data = data

        self._track_type = track_type
        self._with_keypoints = with_keypoints

    @classmethod
    def load(cls, directory: PathLike,
             rig: RiglogData,
             track_type: 'TRACK_TYPE',
             with_keypoints: bool) -> Self:
        #
        svd_path = uglob(directory, '*.npy')
        svd = np.load(svd_path, allow_pickle=True).item()

        #
        if with_keypoints:
            meta_path = uglob(directory, '*.pkl')
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            #
            data_path = uglob(directory, '*.h5')
            data = h5py.File(data_path)['Facemap']
        else:
            meta = None
            data = None

        return FaceMapResult(rig, svd, meta, data, track_type, with_keypoints)

    @classmethod
    def launch_facemap_gui(cls, directory: PathLike,
                           with_keypoints: bool) -> None:
        """GUI view via cli.
        ** Note that launching the GUI required the same source root video path"""
        import subprocess

        svd_path = uglob(directory, '*.npy')

        cmds = ['conda', 'run', '-n', 'rscvp', 'python', '-m', 'facemap']
        cmds.extend(CliArgs('--proc_npy', str(svd_path)).as_command())

        if with_keypoints:
            data_path = uglob(directory, '*.h5')
            cmds.extend(CliArgs('--keypoints', str(data_path)).as_command())

        fprint(f'{cmds=}')
        subprocess.check_call(cmds)

    @property
    def with_keypoints(self) -> bool:
        return self._with_keypoints

    # ============== #
    # Pupil Tracking #
    # ============== #

    def get_pupil_tracking(self) -> PupilDict:
        ret = self.svd['pupil']
        if len(ret) == 1:
            return ret[0]
        raise NotImplementedError('')

    def get_eye_blink(self) -> np.ndarray:
        ret = self.svd['blink']
        if len(ret) == 1:
            return ret[0]
        raise NotImplementedError('')

    # ============== #
    # Frames / Times #
    # ============== #

    @property
    def camera_event(self) -> CamEvent:
        if self._track_type == 'keypoints':
            return self.rig.camera_event['facecam']
        elif self._track_type == 'pupil':
            return self.rig.camera_event['eyecam']
        else:
            raise ValueError('')

    @property
    def fps(self) -> float:
        return self.camera_event.fps

    @property
    def time(self) -> np.ndarray:
        return self.camera_event.time

    @property
    def n_frames(self) -> int:
        return len(self.time)

    # ========= #
    # Keypoints #
    # ========= #

    @property
    def keypoints(self) -> list[KeyPoint]:
        return list(self.data.keys())

    def __getitem__(self, keypoint: KeyPoint) -> KeyPointTrack:
        if keypoint not in self.keypoints:
            raise KeyError(f'{keypoint} invalid')

        x = np.array(self.data[keypoint]['x'])
        y = np.array(self.data[keypoint]['y'])
        llh = np.array(self.data[keypoint]['likelihood'])

        return KeyPointTrack(keypoint, x, y, llh)

    def __iter__(self) -> Generator[KeyPointTrack, None, None]:
        """for all keypoints"""
        for kp in self.keypoints:
            yield self[kp]

    @overload
    def get(self, keypoint: KeyPoint) -> KeyPointTrack:
        """single keypoint"""
        pass

    @overload
    def get(self, keypoint: list[KeyPoint]) -> list[KeyPointTrack]:
        """multiple keypoints"""
        pass

    def get(self, keypoint):
        if isinstance(keypoint, str):
            return self[keypoint]
        elif isinstance(keypoint, list):
            return [self[kp] for kp in keypoint]
        else:
            raise TypeError('')

    def as_array(self, keypoint: list[KeyPoint] | KeyPoint | None = None) -> np.ndarray:
        """(nF, nK)"""
        if keypoint is not None:
            kps = self.get(keypoint)
        else:
            kps = [kp for kp in self]  # all

        #
        if not isinstance(kps, list):
            kps = [kps]

        ret = []
        for kp in kps:  # type: KeyPointTrack
            ret.append(kp.with_outlier_filter().to_zscore().x)

        return np.array(ret).T


# ================= #
# Individual Points #
# ================= #

@attrs.define
class KeyPointTrack:
    name: KeyPoint
    x: np.ndarray
    y: np.ndarray
    likelihood: np.ndarray

    def with_outlier_filter(
            self,
            filter_window: int = 15,
            baseline_window: int = 50,
            max_spike: int = 25,
            max_diff: int = 25
    ) -> Self:
        """x,y with outlier filter

        :param filter_window: window size for median filter
        :param baseline_window: window size for baseline estimation
        :param max_spike: maximum spike size
        :param max_diff: maximum difference between baseline and filtered signal

        """
        from facemap.utils import filter_outliers
        _x, _y = filter_outliers(self.x,
                                 self.y,
                                 filter_window,
                                 baseline_window,
                                 max_spike,
                                 max_diff)

        return attrs.evolve(self, x=_x, y=_y)

    def to_zscore(self) -> Self:
        from scipy.stats import zscore
        _x = zscore(self.x)
        _y = zscore(self.y)
        return attrs.evolve(self, x=_x, y=_y)

