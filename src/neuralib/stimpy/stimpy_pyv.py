from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import final

import attrs
import numpy as np
import polars as pl
from scipy.interpolate import interp1d

from neuralib.util.util_type import PathLike
from neuralib.util.util_verbose import fprint
from .baselog import Baselog, LOG_SUFFIX, StimlogBase, AbstractStimTimeProfile
from .baseprot import AbstractStimProtocol
from .session import Session, SessionInfo
from .stimulus import StimPattern
from .util import try_casting_number, unfold_stimuli_condition

__all__ = ['PyVlog',
           'StimlogPyVStim',
           'PyVProtocol']


@final
class PyVlog(Baselog):
    """class for handle the log file (rig event specific) for pyvstim version (vb lab legacy)"""

    def __init__(self,
                 root_path: PathLike,
                 log_suffix: LOG_SUFFIX = '.log',
                 diode_offset: bool = False):

        super().__init__(root_path, log_suffix, diode_offset)
        self.__prot_cache: PyVProtocol | None = None

    @classmethod
    def _cache_asarray(cls, filepath: Path) -> np.ndarray:
        output = filepath.with_name(filepath.stem + '_log.npy')

        if not output.exists():

            data_list = []
            with filepath.open() as f:
                for line, content in enumerate(f):
                    content = content.strip()
                    if not content.startswith('#') and content != '':  # comments and empty line
                        cols = content.strip().split(',')
                        # Convert the columns to floats
                        cols = [float(x) for x in cols]
                        # Append the row to data_list
                        data_list.append(cols)

            # Find the maximum number of columns
            max_cols = max([len(row) for row in data_list])

            new_data = []

            # Iterate over each row
            for row in data_list:
                # Calculate the number of columns to add
                cols_to_add = max_cols - len(row)
                # Add the required number of np.nan values
                row.extend([np.nan] * cols_to_add)
                # Append the row to new_data
                new_data.append(row)

            # Convert new_data to a numpy array
            ret = np.array(new_data)

            np.save(output, ret)

        return np.load(output)

    # ===== #

    def stimlog_data(self) -> 'StimlogPyVStim':
        return StimlogPyVStim(self)

    def get_prot_file(self) -> PyVProtocol:
        if self.__prot_cache is None:
            self.__prot_cache = PyVProtocol.load(self.stim_prot_file)

        return self.__prot_cache


# ======= #
# Stimlog #
# ======= #

@final
class StimlogPyVStim(StimlogBase):
    """class for handle the log file (stim event specific) for pyvstim version (vb lab legacy)

    `Dimension parameters`:

        N = number of visual stimulation (on-off pairs) = (T * S)

        T = number of trials

        S = number of Stim Type

        P = number of acquisition sample pulse

    """

    v_present_time: np.ndarray
    """(P,) in ms"""
    v_stim: np.ndarray
    """(P,), stim type index. value from 1 to S"""
    v_trial: np.ndarray
    """(P,) number of trial. value from 1 to R. *last is 1, reset?"""
    v_frame: np.ndarray
    """(P,), value 0, 1, 2... TBD"""
    v_blank: np.ndarray
    """(P,). whether is background only. 0: stim display, 1: no stim"""
    v_contrast: np.ndarray
    """(P, ) background contrast, 0 to 1?"""
    v_pos_x: np.ndarray
    """(P, ) display pos x"""
    v_pos_y: np.ndarray
    """(P, ) display pos y"""
    v_ap_x: np.ndarray
    """(P, ) stim center x"""
    v_ap_y: np.ndarray
    """(P, ) stim center y"""
    v_indicator_flag: np.ndarray
    """(P,). photo-indicator, for stim-onset. 1: stim display, 0: no stim"""
    v_duino_time: np.ndarray
    """(P,). extrapolate duinotime from screen indicator. sync arduino time in sec"""

    def __init__(self, riglog: 'PyVlog'):
        super().__init__(riglog, file_path=None)
        self._reset()

    def _reset(self):
        _attrs = (
            'v_present_time',
            'v_stim',
            'v_trial',
            'v_frame',
            'v_blank',
            'v_contrast',
            'v_pos_x',
            'v_pos_y',
            'v_ap_x',
            'v_ap_y',
            'v_indicator_flag'
        )
        code = self.riglog_data.dat[:, 0] == 10
        for i, it in enumerate(self.riglog_data.dat[code, 1:].T):
            setattr(self, f'{_attrs[i]}', it)

        self.v_duino_time = self._get_stim_duino_time(self.riglog_data.dat[code, -1].T)

    def _get_stim_duino_time(self, indicator_flag: np.ndarray) -> np.ndarray:
        """extrapolate duinotime from screen indicator. sync arduino time in (P,) sec"""
        fliploc = np.where(
            np.diff(np.hstack([0, indicator_flag, 0])) != 0
        )[0]

        return interp1d(
            fliploc,
            self.riglog_data.screen_event.time,
            fill_value="extrapolate"
        )(np.arange(len(indicator_flag)))

    @property
    def exp_start_time(self) -> float:
        return self.v_duino_time[0]

    @property
    def stimulus_segment(self) -> np.ndarray:
        return self.get_time_profile().get_time_interval()

    def session_trials(self) -> dict[Session, SessionInfo]:
        raise NotImplementedError('')

    @property
    def time_offset(self) -> float:
        """directly used interpolation using diode signal already"""
        return 0

    def get_stim_pattern(self) -> StimPattern:
        raise NotImplementedError('')

    def exp_end_time(self) -> float:
        return self.v_duino_time[-1]

    # =========== #
    # retinotopic #
    # =========== #

    @property
    def stim_loc(self) -> np.ndarray:
        return np.vstack([self.v_ap_x, self.v_ap_y]).T

    @property
    def avg_refresh_rate(self) -> float:
        """in Hz"""
        return 1 / (np.diff(self.v_duino_time).mean())

    def plot_stim_animation(self):
        from neuralib.plot.animation import plot_scatter_animation
        plot_scatter_animation(self.v_ap_x,
                               self.v_ap_y,
                               self.v_duino_time,
                               step=int(self.avg_refresh_rate))  # TODO check refresh rate?

    def get_time_profile(self) -> StimTimeProfile:
        return StimTimeProfile(self)


@final
@attrs.frozen(repr=False, str=False)
class StimTimeProfile(AbstractStimTimeProfile):
    stim: StimlogPyVStim

    def __repr__(self):
        ret = {
            'n_trials': self.n_trials,
            'foreach_trial_time': np.diff(self.get_time_interval(), axis=1),
            'n_cycle:': self.stim.riglog_data.get_prot_file().get_loops_expr().n_cycles
        }
        return repr(ret)

    __str__ = __repr__

    @property
    def unique_stimuli_set(self) -> pl.DataFrame:
        df = pl.DataFrame({
            'i_stims': self.stim.v_stim.astype(int),
            'i_trials': self.stim.v_trial.astype(int)
        }).unique(maintain_order=True)
        return df

    @property
    def n_trials(self) -> int:
        return self.unique_stimuli_set.shape[0]

    @property
    def i_stim(self) -> np.ndarray:
        """(N, ) value: stim type index"""
        return self.unique_stimuli_set.get_column('i_stims').to_numpy()

    @property
    def i_trial(self) -> np.ndarray:
        """(N, ). value: trial"""
        return self.unique_stimuli_set.get_column('i_trials').to_numpy()

    def get_time_interval(self) -> np.ndarray:
        ustims = self.stim.v_stim * (1 - self.stim.v_blank)
        utrials = self.stim.v_trial * (1 - self.stim.v_blank)

        ret = np.zeros([self.n_trials, 2])
        for i, (st, tr) in enumerate(self.unique_stimuli_set.iter_rows()):
            idx = np.where((ustims == st) & (utrials == tr))[0]
            ret[i, :] = self.stim.v_duino_time[[idx[0], idx[-1]]]

        return ret


# ======== #
# Protocol #
# ======== #

class PyVProtocol(AbstractStimProtocol):
    """
    class for handle the protocol file for pyvstim version (vb lab legacy)

    `Dimension parameters`:

        N = number of visual stimulation (on-off pairs) = (T * S)

        T = number of trials

        S = number of Stim Type

        C = number of Cycle
    """

    @classmethod
    def load(cls, file: Path | str, *,
             cast_numerical_opt=True) -> 'PyVProtocol':

        file = Path(file)
        options = {}
        version = 'pyvstim'

        state = 0
        with Path(file).open() as f:
            for line in f:
                line = line.strip()

                if len(line) == 0 or line.startswith('#'):
                    continue

                if state == 0 and line.startswith('n\t'):
                    header = re.split(r'\t+| +', line)
                    value = [[] for _ in range(len(header))]
                    state = 1

                elif state == 0:
                    idx = line.index('=')
                    if cast_numerical_opt:
                        opt_value = try_casting_number(line[idx + 1:].strip())
                    else:
                        opt_value = line[idx + 1:].strip()

                    options[line[:idx].strip()] = opt_value

                elif state == 1:
                    parts = re.split(r'\t+| +', line, maxsplit=len(header))
                    rows = unfold_stimuli_condition(parts)

                    for r in rows:
                        r.remove('')  # pyvstim interesting problem
                        for i, it in enumerate(r):  # for each col
                            if it != '':
                                value[i].append(it)
                else:
                    raise RuntimeError('illegal state')

            assert len(header) == len(value)
            visual_stimuli = {
                field: value[i]
                for i, field in enumerate(header)
            }

            if 'Shuffle' not in options.keys():
                options['Shuffle'] = False

        return PyVProtocol(file.name, options, pl.DataFrame(visual_stimuli), version)

    @property
    def is_shuffle(self) -> bool:
        """TODO"""
        return False

    @property
    def background(self) -> float:
        """TODO"""
        return self.options.get('background', 0.5)

    @property
    def start_blank_duration(self) -> int:
        raise NotImplementedError('')

    @property
    def blank_duration(self) -> int:
        return self.options['BlankDuration']

    @property
    def trial_blank_duration(self) -> int:
        raise NotImplementedError('')

    @property
    def end_blank_duration(self) -> int:
        raise NotImplementedError('')

    @property
    def trial_duration(self) -> int:
        raise NotImplementedError('')

    @property
    def visual_duration(self) -> int:
        raise NotImplementedError('')

    @property
    def total_duration(self) -> int:
        raise NotImplementedError('')

    def get_loops_expr(self) -> ProtExpression:
        """parse and get the expression and loop number"""
        exprs = []
        n_cycles = []
        n_blocks = self.visual_stimuli_dataframe.shape[0]

        for row in self.visual_stimuli_dataframe.iter_rows():  # each row item_value
            for it in row:
                if isinstance(it, str):
                    if 'loop' in it:
                        match = re.search(r"loop\((.*),(\d+)\)", it)

                        if match:
                            exprs.append(match.group(1))
                            n_cycles.append(match.group(2))
                    else:
                        fprint('loop info not found, check prot file!', vtype='warning')
                        exprs.append('')
                        n_cycles.append(1)

        return ProtExpression(exprs, list(map(int, n_cycles)), n_blocks)


@dataclasses.dataclass
class ProtExpression:
    """
    `Dimension parameters`:

        B = number of block

        C = number of Cycle
    """

    expr: list[str]
    """expression"""
    n_cycles: list[int]
    """number of cycle. length number = B?, value equal to C"""
    n_blocks: int | None
    """number of prot value row (block) B"""

    def __post_init__(self):
        if (len(self.n_cycles) == 2 * self.n_blocks) and self._check_ncycles_foreach_block():
            self.n_cycles = self.n_cycles[::2]
        else:
            raise RuntimeError('')

    def _check_ncycles_foreach_block(self):
        """check if the ncycles are the same and duplicate for each block"""
        n = len(self.n_cycles)
        if n % 2 != 0:
            return False

        for i in range(0, n, 2):
            if self.n_cycles[i] != self.n_cycles[i + 1]:
                return False

        return True
