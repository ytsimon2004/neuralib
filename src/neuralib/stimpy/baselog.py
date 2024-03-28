from __future__ import annotations

import abc
from pathlib import Path
from typing import Literal, TypeVar, Generic

import numpy as np
import polars as pl

from neuralib.stimpy.event import RigEvent, CamEvent
from neuralib.stimpy.session import Session, SessionInfo
from neuralib.stimpy.stimulus import StimPattern

__all__ = [
    'STIMPY_SOURCE_VERSION',
    'LOG_SUFFIX',
    'CAMERA_TYPE',
    #
    'Baselog',
    'StimlogBase',
    'AbstractStimTimeProfile'
]

from neuralib.util.util_type import PathLike

STIMPY_SOURCE_VERSION = Literal['pyvstim', 'stimpy-bit', 'stimpy-git', 'debug']
LOG_SUFFIX = Literal['.log', '.riglog']
CAMERA_TYPE = Literal['facecam', 'eyecam', '1P_cam']

#
R = TypeVar('R', bound='Baselog')  # Riglog-Like
S = TypeVar('S', bound='StimlogBase')  # Stimlog-Like
P = TypeVar('P', bound='AbstractStimProtocol')  # protocol file


# =========== #
# Baselog ABC #
# =========== #

class Baselog(Generic[S, P], metaclass=abc.ABCMeta):
    """Parent ABC class for different stimpy/pyvstim log files. i.e., .log, .riglog"""

    def __init__(self,
                 root_path: PathLike,
                 log_suffix: LOG_SUFFIX,
                 diode_offset: bool = True):
        """

        :param root_path: log file path or log directory
        :param log_suffix: log file suffix
        :param diode_offset: whether do the diode offset
        """

        if not isinstance(root_path, Path):
            root_path = Path(root_path)

        if root_path.is_dir():
            self.riglog_file = self._find_logfile(root_path, log_suffix)
        else:
            self.riglog_file = root_path

        self.version: STIMPY_SOURCE_VERSION = self._check_source_version()
        self.dat = self._cache_asarray(self.riglog_file)

        #
        self._diode_offset = diode_offset

    @classmethod
    def _find_logfile(cls,
                      root: Path,
                      log_suffix: LOG_SUFFIX) -> Path:

        f = list(root.glob(f'*{log_suffix}'))
        if len(f) == 1:
            return f[0]

        elif len(f) == 0:
            print(f'no riglog under {root}, try to find in the subfolder...')
            for s in root.iterdir():
                if s.is_dir() and s.name.startswith('run0'):
                    try:
                        return cls._find_logfile(s, log_suffix)
                    except FileNotFoundError:
                        pass

            raise FileNotFoundError(f'no riglog file {log_suffix} under {root}')

        else:
            raise FileNotFoundError(f'more than one riglog files under {root}')

    def _check_source_version(self) -> STIMPY_SOURCE_VERSION:
        """infer from first line"""
        with open(self.riglog_file) as f:
            for line in f:
                if '#' in line:
                    if 'RIG VERSION' in line:
                        return 'stimpy-bit'
                    elif 'Version' in line:
                        return 'pyvstim'
                    elif 'RIG GIT COMMIT HASH' in line:
                        return 'stimpy-git'
                    else:
                        raise RuntimeError('')

            raise RuntimeError('')

    @classmethod
    @abc.abstractmethod
    def _cache_asarray(cls, filepath: Path) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_prot_file(self) -> P:
        pass

    # ============ #
    # RIGLOG EVENT #
    # ============ #

    def _event(self, code: int) -> np.ndarray:
        """
        :return shape(sample, 2) with time and value
        """
        x = self.dat[:, 0] == code
        t = self.dat[x, 2].copy() / 1000  # s
        v = self.dat[x, 3].copy()
        return np.vstack([t, v]).T

    @property
    def exp_start_time(self) -> float:
        return self.dat[0, 2].copy() / 1000

    @property
    def exp_end_time(self) -> float:
        return self.dat[-1, 2].copy() / 1000

    @property
    def total_duration(self) -> float:
        return self.dat[-1, 2] / 1000

    @property
    def screen_event(self) -> RigEvent:
        return RigEvent('screen', self._event(0))

    @property
    def imaging_event(self) -> RigEvent:
        return RigEvent('imaging', self._event(1))

    @property
    def position_event(self) -> RigEvent:
        return RigEvent('position', self._event(2))

    @property
    def lick_event(self) -> RigEvent:
        return RigEvent('lick', self._event(3))

    @property
    def reward_event(self) -> RigEvent:
        if self.version == 'stimpy-git':
            return RigEvent('reward', self._event(5))
        else:
            return RigEvent('reward', self._event(4))

    @property
    def lap_event(self) -> RigEvent:
        if self.version == 'stimpy-git':
            return RigEvent('lap', self._event(6))
        else:
            return RigEvent('lap', self._event(5))

    def act_event(self) -> np.ndarray:
        pass

    def opto_event(self) -> np.ndarray:
        pass

    class CameraEvent:

        camera: dict[CAMERA_TYPE, int]

        def __init__(self, rig: R):
            self.rig = rig

            if rig.version == 'stimpy-git':
                self.camera = {
                    'facecam': 7,
                    'eyecam': 8,
                    '1P_cam': 9,
                }
            else:
                self.camera = {
                    'facecam': 6,
                    'eyecam': 7,
                    '1P_cam': 8,
                }

        def __getitem__(self, item: CAMERA_TYPE) -> CamEvent:
            if item not in self.camera:
                raise ValueError('cam id not found')
            return CamEvent(item, self.rig._event(self.camera[item]))

    @property
    def camera_event(self) -> CameraEvent:
        return self.CameraEvent(self)

    @property
    def stim_prot_file(self) -> Path:
        return self.riglog_file.with_suffix('.prot')

    @abc.abstractmethod
    def stimlog_data(self) -> S:
        pass


# =========== #
# Stimlog ABC #
# =========== #

class StimlogBase(Generic[R], metaclass=abc.ABCMeta):
    """used for adapted for old (bitbucket master branch) and new (github dev) stimpy

    ** Shape info:
       TR = Numbers of Trial
       ST = Number of Stim Type
       C = Numbers of Cycle
    """

    def __init__(self,
                 riglog: R,
                 file_path: PathLike | None):
        """

        :param riglog:
        :param file_path: filepath of stimlog. could be None if shared log (pyvstim case)
        """
        self.riglog_data = riglog
        if file_path is not None:
            self.stimlog_file = Path(file_path)

    @abc.abstractmethod
    def _reset(self) -> None:
        """used for assign attributes"""
        pass

    @property
    @abc.abstractmethod
    def exp_start_time(self) -> float:
        pass

    @property
    @abc.abstractmethod
    def exp_end_time(self) -> float:
        pass

    @property
    def stim_start_time(self) -> float:
        return self.stimulus_segment[0, 0]

    @property
    def stim_end_time(self) -> float:
        return self.stimulus_segment[-1, 1]

    @property
    @abc.abstractmethod
    def stimulus_segment(self) -> np.ndarray:
        """(N, 2)"""
        pass

    @abc.abstractmethod
    def session_trials(self) -> dict[Session, SessionInfo]:
        pass

    @property
    @abc.abstractmethod
    def time_offset(self) -> float:
        """
        align stimlog time to riglog
        """
        pass

    @abc.abstractmethod
    def get_stim_pattern(self) -> StimPattern:
        pass

    @abc.abstractmethod
    def get_time_profile(self) -> AbstractStimTimeProfile:
        pass


# ================= #
# Stim Time Profile #
# ================= #

class AbstractStimTimeProfile(Generic[S], metaclass=abc.ABCMeta):
    stim: S

    @property
    @abc.abstractmethod
    def unique_stimuli_set(self) -> pl.DataFrame:
        """
        rows: ST * TR;
        col: 2 in i_stim (index of stim type), and i_trials (repetitive trials in the given stim type)"""
        pass

    @property
    @abc.abstractmethod
    def n_trials(self) -> int:
        """nTR"""
        pass

    @abc.abstractmethod
    def get_time_interval(self) -> np.ndarray:
        """(TR, 2) with (start, end)"""
        pass
