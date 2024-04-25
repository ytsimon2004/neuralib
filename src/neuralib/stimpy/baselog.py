from __future__ import annotations

import abc
from pathlib import Path
from typing import Literal, TypeVar, Generic, cast, TypedDict

import numpy as np
import polars as pl

from neuralib.util.util_type import PathLike
from .event import RigEvent, CamEvent
from .session import Session, SessionInfo
from .stimulus import StimPattern

__all__ = [
    'STIMPY_SOURCE_VERSION',
    'LOG_SUFFIX',
    'CAMERA_TYPE',
    #
    'Baselog',
    'StimlogBase',
    'AbstractStimTimeProfile'
]

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

class RigConfig(TypedDict, total=False):
    source_version: STIMPY_SOURCE_VERSION
    """stimpy source version {'pyvstim', 'stimpy-bit', 'stimpy-git'}"""
    version: float
    """acquisition flag. i.e., 0.3"""
    commit_hash: str
    """git commit hash"""
    codes: dict[str, int]
    """<EVENT_TYPES>:<NUMBER>"""
    fields: tuple[str, ...]
    """column repr for the logging"""


class Baselog(Generic[S, P], metaclass=abc.ABCMeta):
    """ABC class for different stimpy/pyvstim log files. i.e., .log, .riglog"""

    log_config: RigConfig

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

        self.log_config = self._get_log_config()
        self.version = self.log_config['source_version']

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

    # noinspection PyTypedDict
    def _get_log_config(self) -> RigConfig:
        """get config dict and source version for different logs from the # headers"""
        ret = RigConfig()
        with open(self.riglog_file) as f:
            for line in f:
                if '#' in line:
                    if 'RIG VERSION' in line:
                        ret['version'] = float(line.split(': ')[-1])

                    elif 'RIG GIT COMMIT HASH' in line:
                        ret['commit_hash'] = line.split(': ')[-1].strip()

                    elif 'CODES' in line:
                        codes = {}
                        content = line.replace('# CODES: ', '').strip()
                        iter_codes = content.split(',')
                        for pair in iter_codes:
                            code, num = pair.split('=')
                            code = code.strip()
                            value = int(num.strip())
                            codes[code.lower()] = value

                        ret['codes'] = codes

                    elif 'RIG CSV' in line:
                        content = line.replace('# RIG CSV: ', '').strip()
                        ret['fields'] = tuple(content.split(','))

        # infer
        if 'opto' not in ret['codes']:
            ret['source_version'] = 'stimpy-bit'

        if 'version' not in ret:
            ret['source_version'] = 'stimpy-git'

        if 'opto' in ret['codes'] and ret['codes']['opto'] == 15:
            ret['source_version'] = 'pyvstim'

        return ret

    @classmethod
    @abc.abstractmethod
    def _cache_asarray(cls, filepath: Path) -> np.ndarray:
        pass

    @abc.abstractmethod
    def get_prot_file(self) -> P:
        """
        get protocol (TypeVar ``P``)

        :return: :class:`~neuralib.stimpy.baseprot.AbstractStimProtocol()`
        """
        pass

    # ============ #
    # RIGLOG EVENT #
    # ============ #

    def _event(self, code: int) -> np.ndarray:
        """shape(sample, 2) with time and value"""
        x = self.dat[:, 0] == code
        t = self.dat[x, 2].copy() / 1000  # s
        v = self.dat[x, 3].copy()
        return np.vstack([t, v]).T

    @property
    def exp_start_time(self) -> float:
        """experimental start time (in sec)"""
        return self.dat[0, 2].copy() / 1000

    @property
    def exp_end_time(self) -> float:
        """experimental end time (in sec)"""
        return self.dat[-1, 2].copy() / 1000

    @property
    def total_duration(self) -> float:
        """experimental duration (in sec)"""
        return self.exp_end_time - self.exp_start_time

    @property
    def screen_event(self) -> RigEvent:
        """screen rig event. i.e., diode pulse"""
        return RigEvent('screen', self._event(0))

    @property
    def imaging_event(self) -> RigEvent:
        """imaging rig event. i.e., 2photon pulse"""
        return RigEvent('imaging', self._event(1))

    @property
    def position_event(self) -> RigEvent:
        """position rig event. i.e., encoder pulse"""
        return RigEvent('position', self._event(2))

    @property
    def lick_event(self) -> RigEvent:
        """lick rig event. i.e., lick meter pulse"""
        return RigEvent('lick', self._event(3))

    @property
    def reward_event(self) -> RigEvent:
        """reward rig event. i.e., reward given pulse from lick meter"""
        if self.version == 'stimpy-git':
            return RigEvent('reward', self._event(5))
        else:
            return RigEvent('reward', self._event(4))

    @property
    def lap_event(self) -> RigEvent:
        """lap rig event. i.e., optic sensing for the reflective taps"""
        if self.version == 'stimpy-git':
            return RigEvent('lap', self._event(6))
        else:
            return RigEvent('lap', self._event(5))

    @property
    def act_event(self) -> RigEvent:
        """todo"""
        raise NotImplementedError('')

    @property
    def opto_event(self) -> RigEvent:
        """todo"""
        raise NotImplementedError('')

    class CameraEvent:
        """camera event"""
        camera: dict[CAMERA_TYPE, int]

        def __init__(self, rig: R):
            """
            :param rig:``Baselog``
            """
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
                raise KeyError('cam id not found')
            return CamEvent(item, self.rig._event(self.camera[item]))

    @property
    def camera_event(self) -> CameraEvent:
        """camera event. including {'facecam', 'eyecam', '1P_cam'} implemented by __getitem__()"""
        return self.CameraEvent(self)

    @property
    def stim_prot_file(self) -> Path:
        """protocol file path"""
        return self.riglog_file.with_suffix('.prot')

    @abc.abstractmethod
    def stimlog_data(self) -> S:
        """get stimlog (TypeVar ``S``)

        :return: :class:`~neuralib.stimpy.baselog.StimlogBase()`
        """
        pass


# =========== #
# Stimlog ABC #
# =========== #

class StimlogBase(Generic[R], metaclass=abc.ABCMeta):
    """ABC for stimulation logging. i.e., .log or .stimlog

    `Dimension parameters`:

        N = number of visual stimulation (on-off pairs) = (T * S)

        T = number of trials

        S = number of Stim Type

        P = number of acquisition sample pulse

    """

    def __init__(self,
                 riglog: R,
                 file_path: PathLike | None):
        """
        :param riglog: :class:`Baselog`
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
        """experimental start time (in sec, synced to riglog time with diode offset)"""
        pass

    @property
    @abc.abstractmethod
    def exp_end_time(self) -> float:
        """experimental end time (in sec, synced to riglog time with diode offset)"""
        pass

    @property
    def stim_start_time(self) -> float:
        """the first stimulation start time (in sec, synced to riglog time with diode offset)"""
        return cast(float, self.stimulus_segment[0, 0])

    @property
    def stim_end_time(self) -> float:
        """the last stimulation end time (in sec, synced to riglog time with diode offset)"""
        return cast(float, self.stimulus_segment[-1, 1])

    @property
    @abc.abstractmethod
    def stimulus_segment(self) -> np.ndarray:
        """stimulation time segment (on-off) in sec (N, 2)"""
        pass

    def stim_square_pulse_event(self, sampling_rate: float = 30.) -> RigEvent:
        """
        Get the stimulation on-off square pulse 0,1 consecutive event

        :param sampling_rate: sampling rate for the time domain interpolation
        :return: Stimulus rig event
        """
        start_time = self.exp_start_time
        end_time = self.exp_end_time
        seg = self.stimulus_segment

        t = np.arange(start_time, end_time, 1 / sampling_rate)
        ret = np.zeros_like(t)
        for (on, off) in seg:
            mask = np.logical_and(on < t, t < off)
            ret[mask] = 1

        return RigEvent('visual_stim', np.vstack((t, ret)).T)

    @abc.abstractmethod
    def session_trials(self) -> dict[Session, SessionInfo]:
        """get the session:SessionInfo dictionary (experimental and user-specific)"""
        pass

    @property
    @abc.abstractmethod
    def time_offset(self) -> float:
        """time (in sec) to sync stimlog time to riglog"""
        pass

    @abc.abstractmethod
    def get_stim_pattern(self) -> StimPattern:
        """get pattern foreach stimulation"""
        pass

    @abc.abstractmethod
    def get_time_profile(self) -> AbstractStimTimeProfile:
        """get time profile"""
        pass


# ================= #
# Stim Time Profile #
# ================= #

class AbstractStimTimeProfile(Generic[S], metaclass=abc.ABCMeta):
    """
    ABC for stimulation time profile

    `Dimension parameters`:

        N = number of visual stimulation (on-off pairs) = (T * S)

        T = number of trials

        S = number of Stim Type

        C = number of Cycle

    """
    stim: S

    @property
    @abc.abstractmethod
    def unique_stimuli_set(self) -> pl.DataFrame:
        """
        rows: number equal to N

        fields: i_stim (index of stim type); i_trials (repetitive trials in the given stim type)
        """
        pass

    @property
    @abc.abstractmethod
    def n_trials(self) -> int:
        """T"""
        pass

    @abc.abstractmethod
    def get_time_interval(self) -> np.ndarray:
        """(N, 2) with (start, end). suppose the same as :attr:`neuralib.stimpy.baselog.StimlogBase.stimulus_segment`"""
        pass
