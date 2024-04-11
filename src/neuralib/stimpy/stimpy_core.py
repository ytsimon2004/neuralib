from __future__ import annotations

import re
from pathlib import Path
from typing import Any, final

import numpy as np
import polars as pl
from typing_extensions import Self

from neuralib.plot.figure import plot_figure
from neuralib.stimpy.baselog import Baselog, LOG_SUFFIX, StimlogBase
from neuralib.stimpy.baseprot import AbstractStimProtocol
from neuralib.stimpy.session import Session, SessionInfo, get_protocol_sessions
from neuralib.stimpy.stimulus import StimPattern
from neuralib.stimpy.util import unfold_stimuli_condition, try_casting_number
from neuralib.util.util_type import PathLike
from neuralib.util.util_verbose import fprint, printdf
from neuralib.util.utils import deprecated

__all__ = ['RiglogData',
           'StimpyProtocol']


@final
class RiglogData(Baselog):
    def __init__(self,
                 root_path: PathLike,
                 log_suffix: LOG_SUFFIX = '.riglog',
                 diode_offset: bool = True):

        super().__init__(root_path, log_suffix, diode_offset)

        #
        self.__stimlog_cache: StimlogBase | None = None
        self.__prot_cache: StimpyProtocol | None = None

    @classmethod
    def _cache_asarray(cls, filepath: Path) -> np.ndarray:
        output = filepath.with_name(filepath.stem + '_riglog.npy')

        if not output.exists():
            riglog = np.loadtxt(
                filepath,
                delimiter=',',
                comments=['#', 'None'],
                converters={
                    0: lambda it: float(it[1:]),  # it: parameter name: the text of the first column
                    3: lambda it: float(it[:-1])
                }
            )
            np.save(output, riglog)

        return np.load(output)

    @deprecated(reason='caller rare')
    def session_handler(self, session: str | tuple[str, ...]) -> tuple[float, float]:
        """trigger `select_time_range` for specific time range"""
        strial = self.stimlog_data().session_trials()
        if isinstance(session, str):
            t0, t1 = strial[session].time
        elif isinstance(session, tuple):
            t_all = [strial[s].time for s in session]
            t0, t1 = np.min(t_all), np.max(t_all)
            fprint(f'get trange from multiple sessions, {session}: t0:{t0}, t1:{t1}')
        else:
            raise TypeError(f'{type(session)}')

        self.select_time_range(t0, t1)

        return t0, t1

    def select_time_range(self, t0: float, t1: float):
        """overwrite the `instance attr: data` to given time range"""
        t = self.dat[:, 2] / 1000
        mask = np.logical_and(t0 < t, t < t1)
        self.dat = self.dat[mask]

    @property
    def stimlog_file(self) -> Path:
        return self.riglog_file.with_suffix('.stimlog')

    def stimlog_data(self) -> StimlogBase:
        if self.__stimlog_cache is None:
            if self.version == 'stimpy-git':
                from .stimpy_git import StimlogGit
                self.__stimlog_cache = StimlogGit(self, self.stimlog_file, self._diode_offset)
            elif self.version == 'stimpy-bit':
                self.__stimlog_cache = Stimlog(self, self.stimlog_file, self._diode_offset)
            else:
                raise ValueError(f'unknown version: {self.version}')

        return self.__stimlog_cache

    def get_prot_file(self) -> StimpyProtocol:
        if self.__prot_cache is None:
            self.__prot_cache = StimpyProtocol.load(self.stim_prot_file)

        return self.__prot_cache


@final
class Stimlog(StimlogBase):
    # vstim: start from code 10 in .stimlog
    v_present_time: np.ndarray
    v_stim: np.ndarray
    v_trial: np.ndarray
    v_photo: np.ndarray
    v_contrast: np.ndarray
    v_ori: np.ndarray
    v_sf: np.ndarray
    v_phase: np.ndarray
    v_stim_idx: np.ndarray

    # state machine: start from code 20 in .stimlog
    s_on_v: np.ndarray
    s_present_time: np.ndarray
    s_cycle: np.ndarray
    s_new_state: np.ndarray
    s_old_state: np.ndarray
    s_state_elapsed: np.ndarray
    s_trial_type: np.ndarray

    def __init__(self, riglog: RiglogData,
                 file_path: PathLike,
                 diode_offset: bool = True,
                 sequential_offset: bool = True):

        super().__init__(riglog, file_path)
        self.diode_offset = diode_offset

        # diode
        self._do_sequential_diode_offset = sequential_offset
        self._cache_time_offset: float | np.ndarray | None = None

        self._reset()

    def _reset(self):
        v_present_time = []
        v_stim = []
        v_trial = []
        v_photo = []
        v_contrast = []
        v_ori = []
        v_sf = []
        v_phase = []
        v_stim_idx = []
        s_on_v = []
        s_present_time = []
        s_cycle = []
        s_new_state = []
        s_old_state = []
        s_state_elapsed = []
        s_trial_type = []

        with self.stimlog_file.open() as _f:
            for number, line in enumerate(_f):
                line = line.strip()  # remove the space at the beginning and the end of the string
                if len(line) == 0 or line.startswith('#'):
                    continue
                part = line.split(',')

                try:
                    code = int(part[0])

                    if code == 10:
                        v_present_time.append(float(part[1]))

                        try:
                            stim_value = int(part[2])
                        except ValueError:  # 'None'
                            stim_value = -1
                        v_stim.append(stim_value)

                        v_trial.append(int(part[3]))
                        v_photo.append(int(part[4]))

                        if len(part) > 5:
                            v_contrast.append(float(part[5]))
                            v_ori.append(int(float(part[6])))
                            v_sf.append(float(part[7]))
                            v_phase.append(float(part[8]))
                            v_stim_idx.append(int(part[9]))
                        else:
                            # empty list item = -1
                            v_contrast.append(-1)
                            v_ori.append(-1)
                            v_sf.append(-1)
                            v_phase.append(-1)
                            v_stim_idx.append(-1)

                    elif code == 20:
                        s_on_v.append(v_present_time[-1])  # sync s to v
                        s_present_time.append(int(part[1]) / 1000)
                        s_cycle.append(int(part[2]))
                        s_new_state.append(int(part[3]))
                        s_old_state.append(int(part[4]))
                        s_state_elapsed.append(int(part[5]) / 1000)
                        s_trial_type.append(int(part[6]))

                    else:
                        raise ValueError(f'unknown code : {code}')

                except BaseException as e:
                    raise RuntimeError(f'line {number}: {line}') from e

        self.v_present_time = np.array(v_present_time)
        self.v_stim = np.array(v_stim)
        self.v_trial = np.array(v_trial)
        self.v_photo = np.array(v_photo)
        self.v_contrast = np.array(v_contrast)
        self.v_ori = np.array(v_ori)
        self.v_sf = np.array(v_sf)
        self.v_phase = np.array(v_phase)
        self.v_stim_idx = np.array(v_stim_idx)

        self.s_on_v = np.array(s_on_v)
        self.s_present_time = np.array(s_present_time)
        self.s_cycle = np.array(s_cycle)
        self.s_new_state = np.array(s_new_state)
        self.s_old_state = np.array(s_old_state)
        self.s_state_elapsed = np.array(s_state_elapsed)
        self.s_trial_type = np.array(s_trial_type)

    @property
    def exp_start_time(self) -> float:
        tstart = self.v_present_time[0]

        if isinstance(self.time_offset, float):
            return tstart + self.time_offset
        elif isinstance(self.time_offset, np.ndarray):
            return tstart + self.time_offset[0]

    @property
    def exp_end_time(self) -> float:
        tend = self.v_present_time[-1]

        if isinstance(self.time_offset, float):
            return tend + self.time_offset
        elif isinstance(self.time_offset, np.ndarray):
            return tend + self.time_offset[-1]

    @property
    def stim_start_time(self) -> float:
        v_start = np.nonzero(self.v_stim_idx == 1)[0][0]
        tstart = self.v_present_time[v_start]

        if isinstance(self.time_offset, float):
            return tstart + self.time_offset
        elif isinstance(self.time_offset, np.ndarray):
            return tstart + self.time_offset[0]

    @property
    def stim_end_time(self) -> float:
        v_end = np.nonzero(np.diff(self.v_stim_idx) < 0)[0][-1] + 1
        tend = self.v_present_time[v_end]

        if isinstance(self.time_offset, float):
            return tend + self.time_offset
        elif isinstance(self.time_offset, np.ndarray):
            return tend + self.time_offset[-1]

    @property
    def stimulus_segment(self) -> np.ndarray:
        """sti_period"""
        v_start = self.v_stim_idx == 1
        t1 = self.v_present_time[v_start]
        t2 = self.v_present_time[np.nonzero(np.diff(self.v_stim_idx) < 0)[0] + 1]

        if isinstance(self.time_offset, float):
            offset = self.time_offset
        elif isinstance(self.time_offset, np.ndarray):
            offset = np.stack([self.time_offset, self.time_offset], axis=1)
        else:
            raise TypeError('')

        t = np.vstack((t1, t2)).T + offset

        return t

    @property
    def time_offset(self) -> float | np.ndarray:
        if self._cache_time_offset is None:
            self._cache_time_offset = diode_time_offset(self.riglog_data,
                                                        self.diode_offset,
                                                        return_sequential=self._do_sequential_diode_offset)
        return self._cache_time_offset

    def session_trials(self) -> dict[Session, SessionInfo]:
        """get session dict"""

        return {
            prot.name: prot
            for prot in get_protocol_sessions(self)
        }

    def get_stim_pattern(self) -> StimPattern:
        prot = StimpyProtocol.load(self.stimlog_file.with_suffix('.prot'))
        v_start = self.v_stim_idx == 1
        t = self.stimulus_segment
        dire = self.v_ori[v_start]
        sf = self.v_sf[v_start]
        contrast = self.v_contrast[v_start]
        nr = self.v_stim[v_start]  # sti_nr = 0-71, len: 360

        tf = prot.tf[nr]
        dur = prot['dur'][nr]

        return StimPattern(t, dire, sf, tf, contrast, dur)

    def get_time_profile(self):
        raise NotImplementedError('')


# ================= #
# Diode Time Offset #
# ================= #

class DiodeNumberMismatchError(ValueError):

    def __init__(self):
        super().__init__('Diode numbers are not detected reliably')


class DiodeSignalMissingError(RuntimeError):

    def __init__(self):
        super().__init__('no diode signal were found')


def diode_time_offset(rig: RiglogData,
                      diode_offset: bool = True,
                      return_sequential: bool = True,
                      default_offset_value: float = 0.6) -> float | np.ndarray:
    """
    time offset used in the `old stimpy`
    offset time from screen_time .riglog (diode) to .stimlog
    ** normally stimlog time value are smaller than riglog

    :param rig:
    :param diode_offset: whether correct diode signal
    :param return_sequential: return sequential offset, if False, use mean value across diode pulses
    :param default_offset_value: hardware(rig)-dependent offset value

    :return: tuple of offset average and std value
    """

    stimlog = rig.stimlog_data()
    if not isinstance(stimlog, Stimlog):
        raise TypeError('')

    if not diode_offset:
        fprint('no offset', vtype='warning')
        return default_offset_value

    #
    try:
        t = _diode_offset_sequential(rig, debug_plot=False)
    except DiodeNumberMismatchError as e:
        try:
            first_pulse = _check_if_diode_pulse(rig)
            fprint(f'{repr(e)}, use the first pulse diff for alignment', vtype='warning')
            return first_pulse

        except DiodeSignalMissingError as e:
            fprint(f'{repr(e)}, use default value', vtype='warning')
            return default_offset_value

    #
    avg_t = float(np.mean(t))
    std_t = float(np.std(t))
    if not (0 <= avg_t <= 1):
        fprint(f'{avg_t} too large, might not be properly calculated, check...', vtype='warning')

    fprint(f'avg: {avg_t}, std: {std_t}')

    if return_sequential:
        return t
    else:
        return avg_t


def _check_if_diode_pulse(rig: RiglogData) -> float:
    """only count 1st difference"""
    screen_time = rig.screen_event.time
    stimlog = rig.stimlog_data()
    if not isinstance(stimlog, Stimlog):
        raise TypeError('')
    stimlog_vstart = stimlog.v_present_time[(stimlog.v_stim_idx == 1)]

    try:
        # noinspection PyTypeChecker
        return screen_time[0] - stimlog_vstart[0]
    except IndexError as e:
        raise DiodeSignalMissingError() from e


def _diode_offset_sequential(rig: RiglogData, debug_plot: bool = False) -> np.ndarray:
    stimlog = rig.stimlog_data()
    if not isinstance(stimlog, Stimlog):
        raise TypeError('')

    stimlog_vstart = stimlog.v_present_time[(stimlog.v_stim_idx == 1)]
    riglog_vstart = rig.screen_event.time[0::2]

    if len(riglog_vstart) != len(stimlog_vstart):
        raise DiodeNumberMismatchError()
    else:
        if debug_plot:
            _plot_time_alignment_diode(riglog_vstart, stimlog_vstart)

        return riglog_vstart - stimlog_vstart


def _plot_time_alignment_diode(riglog_screen: np.ndarray,
                               stimlog_time: np.ndarray):
    """Plot time alignment (stimlog time value smaller than riglog)"""
    with plot_figure(None) as ax:
        ax.plot(riglog_screen - stimlog_time)
        ax.set(xlabel='Visual stim #', ylabel='Time diff (s)')


# ======== #
# Protocol #
# ======== #

class StimpyProtocol(AbstractStimProtocol):
    """Stimpy protocol file."""

    def __repr__(self):
        ret = list()

        ret.append('# general parameters')
        for k, v in self.options.items():
            ret.append(f'{k} = {v}')
        ret.append('# stimulus conditions')
        ret.append('\t'.join(self.stim_params))

        ret.append(printdf(self.visual_stimuli_dataframe))

        return '\n'.join(ret)

    @classmethod
    def load(cls, file: PathLike) -> Self:
        """Load *.prot file.

        :param file:
        :return:
        """
        file = Path(file)
        options = {}
        version = 'stimpy-bit'

        state = 0
        with file.open() as f:
            for line in f:
                line = line.strip()

                if len(line) == 0 or line.startswith('#'):
                    continue

                # change state to 1 if # stimulus conditions
                if state == 0 and line.startswith('n '):
                    header = re.split(' +', line)  # extract header
                    data = [[] for _ in range(len(header))]
                    state = 1

                elif state == 0:
                    idx = line.index('=')
                    options[line[:idx].strip()] = try_casting_number(line[idx + 1:].strip())

                elif state == 1:
                    parts = re.split(' +', line, maxsplit=len(header))
                    rows = unfold_stimuli_condition(parts)

                    if len(rows) != 1:
                        version = 'stimpy-git'  # github stimpy

                    for r in rows:
                        for i, it in enumerate(r):  # for each col
                            data[i].append(it)
                else:
                    raise RuntimeError('illegal state')

        assert len(header) == len(data)

        visual_stimuli = {
            field: list(map(str, data[i])) if field == 'evolveParams' else data[i]
            for i, field in enumerate(header)
        }

        # noinspection PyTypeChecker
        return StimpyProtocol(file.name, options, pl.DataFrame(visual_stimuli), version)

    @property
    def controller(self) -> str:
        """protocol controller name"""
        return self.options['controller']

    @property
    def is_shuffle(self) -> bool:
        """Is stimulus ordering shuffled"""
        return self.options['shuffle'] == 'True'

    @property
    def background(self) -> float:
        return float(self.options.get('background', 0.5))

    @property
    def start_blank_duration(self) -> int:
        return self.options.get('startBlankDuration', 5)

    @property
    def blank_duration(self) -> int:
        return self.options['blankDuration']

    @property
    def trial_blank_duration(self) -> int:
        return 2  # TODO not tested/checked value?

    @property
    def end_blank_duration(self) -> int:
        return self.options.get('endBlankDuration', 5)

    @property
    def trial_duration(self) -> int:
        dur = self['dur']
        return np.sum(dur) + len(dur) * self.blank_duration + self.trial_blank_duration

    @property
    def visual_duration(self) -> int:
        return self.trial_duration * self.n_trials

    @property
    def total_duration(self) -> int:
        return self.start_blank_duration + self.visual_duration + self.end_blank_duration

    @property
    def texture(self) -> str:
        """stimulus texture, {circle, sqr... todo check in stimpy}"""
        return self.options['texture']

    @property
    def mask(self) -> str:
        """todo check in stimpy"""
        return self.options['mask']

    class EvoledParameter(dict[str, Any]):

        def __init__(self, keys: list[str], data: np.ndarray):
            self.__keys = keys
            self.__data = data  # {'phase':['linear',1]}

        def __len__(self) -> int:
            return len(self.__keys)

        def keys(self):
            return set(self.__keys)

        def __contains__(self, o) -> bool:
            return o in self.__keys

        def __getitem__(self, k: str) -> Any:
            if k not in self.__keys:  # i.e., 'phase'
                raise KeyError()

            # TODO only work for 'linear' case
            ret = []
            for d in self.__data:
                ret.append(d[k][1])

            return np.array(ret)

    def evolve_param_headers(self) -> list[str]:
        """Get parameter header which set in the 'evolveParams, i.g., 'phase'

        :return: header list.

        """
        keys = set()
        data = self['evolveParams']
        for d in data:
            d = eval(d)
            keys.update(d.keys())
        return list(keys)

    @property
    def evolve_params(self) -> EvoledParameter:
        """Get value from 'evolveParams'.

        Examples:

        >>> log: StimpyProtocol

        How many parameters in evolveParams

        >>> len(log.evolve_params)

        List parameter in 'evolveParams'

        >>> log.evolve_params.keys()

        Dose parameter 'phase' in 'evolveParams'?

        >>> 'phase' in log.evolve_params

        Get phase value from 'evolveParams'.

        >>> log.evolve_params['phase']

        :return:
        """
        if self.version == '2022':
            raise DeprecationWarning('new stimpy has no evolveParams header')

        data = np.array([eval(it) for it in self['evolveParams']])  # cast back to dict
        return self.EvoledParameter(self.evolve_param_headers(), data)

    @property
    def tf(self) -> np.ndarray:
        if self.version == 'stimpy-bit':
            return self.evolve_params['phase']
        elif self.version == 'stimpy-git':
            return self['tf']
        else:
            raise NotImplementedError('')

    def to_dict(self) -> dict[str, Any]:
        ret = {
            'controller': self.controller,
            'displayType': self.options['displayType'],
            'background': self.options['background'],
            'stimulusType': self.options['stimulusType'],
            'nTrials': self.n_trials,
            'shuffle': self.options['shuffle'],
            'blankDuration': self.blank_duration,
            'startBlankDuration': self.start_blank_duration,
            'endBlankDuration': self.end_blank_duration,
            'visual_stimuli': self.visual_stimuli_dataframe
        }

        return ret
