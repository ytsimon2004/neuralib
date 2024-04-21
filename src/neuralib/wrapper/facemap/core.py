from __future__ import annotations

import pickle
from typing import TypedDict, Generator, overload, final, Final, Literal

import attrs
import h5py
import numpy as np
from typing_extensions import TypeAlias, Self

from neuralib.util.cli_args import CliArgs
from neuralib.util.util_type import PathLike
from neuralib.util.util_verbose import fprint
from neuralib.util.utils import uglob

__all__ = [
    'TRACK_TYPE',
    'SVDVariables',
    #
    'KeyPoint',
    'FaceMapResult',
    'KeyPointTrack',
]

TRACK_TYPE = Literal['keypoints', 'pupil']


class PupilDict(TypedDict):
    """
    `Dimension parameters`:

        F: number pf frames
    """
    area: np.ndarray
    """(F,)"""
    com: np.ndarray
    """center of maze (F, 2)"""
    axdir: np.ndarray
    """(F, 2, 2)"""
    axlen: np.ndarray
    """(F, 2)"""
    area_smooth: np.ndarray
    """(F,)"""
    com_smooth: np.ndarray
    """(F, 2)"""


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
    running: list[np.ndarray]  # TODO check
    blink: list[np.ndarray]
    rois: list[RoiDict]  # TODO check
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


# ============== #
# FaceMap Result #
# ============== #

KeyPoint: TypeAlias = str


@final
class FaceMapResult:
    """facemap result container

    `Dimension parameters`:

        F = number of video frames

        K = number of keypoints

    """
    __slots__ = ('svd', 'meta', 'data', 'frame_time',
                 'track_type', 'with_keypoints')

    def __init__(
            self,
            svd: SVDVariables | None,
            meta: KeyPointsMeta | None,
            data: h5py.Group | None,
            track_type: TRACK_TYPE,
            with_keypoints: bool,
    ):
        """
        :param svd: SVD processing outputs
        :param meta: Optional for Keypoints processing (result)
        :param data: Optional for Keypoints processing (config)
        :param track_type: {'keypoints', 'pupil'}
        :param with_keypoints: if has keypoint tracking result
        """
        self.svd: Final[SVDVariables | None] = svd
        self.meta: Final[KeyPointsMeta | None] = meta
        self.data: Final[h5py.Group | None] = data

        self.track_type: Final[TRACK_TYPE] = track_type
        self.with_keypoints: Final[bool] = with_keypoints

    @classmethod
    def load(cls, directory: PathLike,
             track_type: TRACK_TYPE,
             *,
             file_pattern: str = '') -> Self:
        """
        Load the facecam result from its output directory

        :param directory: directory contains the possible facemap output files (`*.npy`, `*.pkl`, and `*.h5`)
        :param track_type: {'keypoints', 'pupil'}
        :param file_pattern: string prefix pattern to glob the facemap output file
        :return: :class:`FaceMapResult`
        """
        #
        try:
            svd_path = uglob(directory, f'{file_pattern}*.npy')
        except FileNotFoundError:
            svd = None
        else:
            svd = np.load(svd_path, allow_pickle=True).item()

        #
        try:
            meta_path = uglob(directory, f'{file_pattern}*.pkl')
        except FileNotFoundError:
            meta = None
            data = None
            with_keypoints = False
        else:
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)

            data_path = uglob(directory, f'{file_pattern}*.h5')
            data = h5py.File(data_path)['Facemap']

            with_keypoints = True

        cls._verify_data_content(directory, track_type, svd, data)

        return FaceMapResult(svd, meta, data, track_type, with_keypoints)

    @classmethod
    def _verify_data_content(cls, directory, track_type, svd, data):
        if track_type == 'keypoints' and data is None:
            raise RuntimeError(f'.h5 data file not found in the {directory}')
        elif track_type == 'pupil' and svd is None:
            raise RuntimeError(f'.npy svd file not found in the {directory}')

    @classmethod
    def launch_facemap_gui(cls, directory: PathLike,
                           with_keypoints: bool,
                           *,
                           file_pattern: str = '',
                           env_name: str = 'neuralib') -> None:
        """
        GUI view via cli.

        **Note that calling this method will overwrite `filenames` field in *proc.npy**

        :param directory: directory contains the possible facemap output files (`*.npy`, `*.pkl`, and `*.h5`),
            and also the raw video file
        :param with_keypoints: if has keypoint tracking result
        :param file_pattern: string prefix pattern to glob the facemap output file and raw avi file
        :param env_name: conda env name that installed the facemap package
        """
        import subprocess

        cls._modify_video_filenames_field(directory)

        svd_path = uglob(directory, f'{file_pattern}*.npy')

        cmds = ['conda', 'run', '-n', f'{env_name}', 'python', '-m', 'facemap']
        cmds.extend(CliArgs('--proc_npy', str(svd_path)).as_command())

        if with_keypoints:
            data_path = uglob(directory, '*.h5')
            cmds.extend(CliArgs('--keypoints', str(data_path)).as_command())

        fprint(f'{cmds=}')
        subprocess.check_call(cmds)

    @classmethod
    def _modify_video_filenames_field(cls,
                                      directory: PathLike, *,
                                      file_pattern: str = ''):
        """brute force rewrite ``filenames`` field in raw file"""
        svd_path = uglob(directory, f'{file_pattern}*.npy')
        video_path = uglob(directory, f'{file_pattern}*.avi')

        dat = np.load(svd_path, allow_pickle=True).item()
        dat['filenames'] = [[str(video_path)]]
        np.save(svd_path, dat, allow_pickle=True)
        fprint(f'overwrite filenames field to {str(video_path)}', vtype='warning')

    # ============== #
    # Pupil Tracking #
    # ============== #

    def get_pupil_tracking(self) -> PupilDict:
        """pupil tracking result"""
        ret = self.svd['pupil']
        if len(ret) == 1:
            return ret[0]
        raise NotImplementedError('')

    def get_pupil_area(self) -> np.ndarray:
        return self.get_pupil_tracking()['area_smooth']

    def get_eye_blink(self) -> np.ndarray:
        """eye blinking array (F, )"""
        ret = self.svd['blink']
        if len(ret) == 1:
            return ret[0]
        raise NotImplementedError('')

    # ========= #
    # Keypoints #
    # ========= #

    @property
    def keypoints(self) -> list[KeyPoint]:
        """list of keypoint name"""
        return list(self.data.keys())

    def __getitem__(self, keypoint: KeyPoint) -> KeyPointTrack:
        """get a specific keypoint tracking result"""
        if keypoint not in self.keypoints:
            raise KeyError(f'{keypoint} invalid, please select from {self.keypoints}')

        x = np.array(self.data[keypoint]['x'])
        y = np.array(self.data[keypoint]['y'])
        llh = np.array(self.data[keypoint]['likelihood'])

        return KeyPointTrack(keypoint, x, y, llh)

    def __iter__(self) -> Generator[KeyPointTrack, None, None]:
        """iterate all keypoints"""
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
        """get a single or multiple keypoint(s)"""
        if isinstance(keypoint, str):
            return self[keypoint]
        elif isinstance(keypoint, list):
            return [self[kp] for kp in keypoint]
        else:
            raise TypeError('')

    def as_array(self,
                 keypoint: list[KeyPoint] | KeyPoint | None = None,
                 with_outlier_filter: bool = True,
                 to_zscore: bool = True,
                 **kwargs) -> np.ndarray:
        """
        get keypoint(s) result as an 2D array with shape (K, F, 2). 3rd dim indicates the xy

        :param keypoint: keypoint
        :param with_outlier_filter:
        :param to_zscore:
        :param kwargs: pass through :meth:`~KeyPointTrack.with_outlier_filter()`
        :return: (K, F, 2)
        """
        if keypoint is not None:
            kps = self.get(keypoint)
        else:
            kps = [kp for kp in self]  # all

        #
        if not isinstance(kps, list):
            kps = [kps]

        ret = []
        for kp in kps:  # type: KeyPointTrack

            if with_outlier_filter:
                kp = kp.with_outlier_filter(**kwargs)

            if to_zscore:
                kp = kp.to_zscore()

            ret.append(np.vstack([kp.x, kp.y]).T)

        return np.array(ret)


# ================= #
# Individual Points #
# ================= #

@attrs.define
class KeyPointTrack:
    """single keypoint tracked result"""

    name: KeyPoint
    """name of keypoint"""
    x: np.ndarray
    """x loc (F,)"""
    y: np.ndarray
    """y loc (F,)"""
    likelihood: np.ndarray
    """tracking likelihood (F,)"""

    @property
    def mean_xy(self) -> np.ndarray:
        """mean x y loc"""
        return (self.x + self.y) / 2

    def with_outlier_filter(
            self,
            filter_window: int = 15,
            baseline_window: int = 50,
            max_spike: int = 25,
            max_diff: int = 25
    ) -> Self:
        """x,y with outlier filter (remove jump and do the interpolation)

        :param filter_window: window size for median filter
        :param baseline_window: window size for baseline estimation
        :param max_spike: maximum spike size
        :param max_diff: maximum difference between baseline and filtered signal
        :return: :class:`KeyPointTrack`

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
        """x, y z-scoring"""
        from scipy.stats import zscore
        _x = zscore(self.x)
        _y = zscore(self.y)
        return attrs.evolve(self, x=_x, y=_y)
