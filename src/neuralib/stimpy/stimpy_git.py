from __future__ import annotations

import re
from typing import Any, Callable, final

import numpy as np
import polars as pl

from neuralib.util.util_type import PathLike
from neuralib.util.util_verbose import fprint
from neuralib.util.utils import deprecated
from .baselog import StimlogBase
from .session import Session, SessionInfo, get_protocol_sessions
from .stimpy_core import RiglogData, StimpyProtocol
from .stimulus import StimPattern

__all__ = ['StimlogGit']


# TODO code 0 options need to be extended
@final
class StimlogGit(StimlogBase):
    """class for handle the stimlog file for stimpy **github** version
    (mainly tested in the commits derived from master branch)

    `Dimension parameters`:

        N = number of visual stimulation (on-off pairs) = (T * S)

        T = number of trials

        S = number of Stim Type

        V = number of acquisition sample pulse (Visual parameters)

        I = number of photo indicator pulse
    """

    log_name: str | None
    code_version: str | None
    code_tags: str | None
    header: list[str]
    rig_trigger: tuple[str, float]
    start_time: float
    end_time: float
    missing_frame: int

    log_info: dict[int, str]
    """{0: <'Gratings', ...>, 1: 'PhotoIndicator', 2: 'StateMachine', 3: 'LogDict'}"""
    log_header: dict[int, list[str]]
    """list of header"""
    log_data: dict[int, list[tuple[Any, ...]]]
    """only for StateMachine(2) and LogDict(3)"""

    # Visual Presentation <Grating, ...>
    v_present_time: np.ndarray  # (V,) float
    v_duration: np.ndarray  # (V,)
    v_contrast: np.ndarray  # (V,)
    v_ori: np.ndarray  # (V,)
    v_phase: np.ndarray  # (V,)
    v_pos: np.ndarray  # shape(V, 2)
    v_size: np.ndarray  # shape(V, 2)
    v_flick: np.ndarray  # (V,)
    v_interpolate: np.ndarray  # (V,) bool
    v_mask: np.ndarray  # (V, ) bool, TODO
    v_sf: np.ndarray  # (V,)
    v_tf: np.ndarray  # (V,)
    v_opto: np.ndarray  # (V,) int
    v_pattern: np.ndarray

    # PhotoIndicator
    p_time: np.ndarray  # (I,)
    p_state: np.ndarray  # (I,) bool
    p_size: np.ndarray  # (I,)
    p_pos: np.ndarray  # (I, 2)
    p_units: np.ndarray  # (I,)  str
    p_mode: np.ndarray  # (I,)  int
    p_frames: np.ndarray  # (I,)  int
    p_enabled: np.ndarray  # (I,)  bool

    def __init__(self, riglog: RiglogData,
                 file_path: PathLike,
                 diode_offset=True):

        super().__init__(riglog, file_path)
        self.diode_offset = diode_offset
        self._cache_time_offset: float | None = None

        self._reset()

    def _reset(self):
        self.log_name = None
        self.code_version = None
        self.code_tags = None
        self.header = []
        self.rig_trigger: tuple[str, float] = ('', 0)
        self.start_time = np.nan  # TODO check, use for sync?
        self.end_time = np.nan  # TODO check
        self.missing_frame = 0
        self.log_info = {}
        self.log_header = {}
        log_data = {}

        with self.stimlog_file.open() as f:
            for line, content in enumerate(f):  # type: int, str
                content = content.strip()
                # `## (CODE VERSION) : (commit hash: 88c4705 - tags: [''])`
                # *: 0-inf; +: 1-inf; ?: 0-1
                m = re.match(r'#+ (.+?)\s*:\s*(.+)', content)
                if m:
                    info_name = m.group(1)
                    info_value = m.group(2)

                    if info_name == 'LOG NAME':
                        self.log_name = info_value
                    elif info_name == 'CODE VERSION':
                        self._reset_code_version(info_value)
                    elif info_name == 'Format':
                        self.header = info_value.split(' ')
                        if self.header != ['source_id', 'time', 'source_infos']:
                            raise RuntimeError('stimlog format changed')
                    elif info_name == 'Rig trigger on':
                        info_value = info_value.split(',')
                        self.rig_trigger = (info_value[0], float(info_value[1]))
                    elif self._reset_log_info(info_name, info_value):
                        pass
                    else:
                        print(f'ignore header line at {line + 1} : {content}')

                elif content.startswith('### START'):
                    self.start_time = float(content[9:].strip())
                elif content.startswith('### END'):
                    self.end_time = float(content[7:].strip())
                elif content.startswith('# Missed') and content.endswith('frames'):
                    self.missing_frame = int(content.split(' ')[2])
                elif content.startswith('#'):
                    print(f'ignore header line at {line + 1} : {content}')

                else:
                    try:
                        self._reset_line(log_data, line + 1, content)
                    except BaseException:
                        raise RuntimeError(f'bad format line at {line + 1} : {content}')

        self.log_data = self._reset_final(log_data)

    def _reset_code_version(self, info_value: str):
        """for ### CODE VERSION,  commit hash and tags"""
        info_value = info_value.strip().split(' ')
        self.code_version = info_value[info_value.index('hash:') + 1]
        self.code_tags = info_value[info_value.index('tags:') + 1]
        if self.code_version == 'None':
            self.code_version = None
        if self.code_tags == 'None':
            self.code_tags = None

    def _reset_log_info(self, code: str, info_value: str):
        try:
            code = int(code)
        except ValueError:
            return False

        name, header = info_value.split(' ', maxsplit=1)
        header = eval(header)
        self.log_info[code] = name
        self.log_header[code] = header
        return True

    def _reset_line(self, log_data: dict[int, list], line: int, content: str):
        message: str
        value: list
        code, time, message = content.split(' ', maxsplit=2)
        code = int(code)
        time = float(time)

        if self.log_info[code] in ('Gratings', 'FunctionBased', 'PhotoIndicator', 'LogDict'):
            value = eval(message)
            if len(self.log_header[code]) != len(value):
                print(f'log category {code} size mismatched at line {line} : {message}')
            else:
                log_data.setdefault(code, []).append((time, *value))

        elif self.log_info[code] == 'StateMachine':
            # [<States.SHOW_STIM: 2>, <States.SHOW_BLANK: 1>]
            value = eval(message.replace('<', '("').replace(':', '",').replace('>', ')'))
            # [("States.SHOW_STIM", 2), ("States.SHOW_BLANK", 1)]
            if len(self.log_header[code]) != len(value):
                print(f'log category {code} size mismatched at line {line} : {message}')
            else:
                log_data.setdefault(code, []).append((time, *value))

        else:
            print(f'unknown log category : at line {line} : {content}')

    def _reset_final(self, log_data: dict[int, list]) -> dict[int, list]:
        remove_code = []
        for code, content in log_data.items():
            if self.log_info[code] in ('Gratings', 'FunctionBased'):
                self.v_present_time = np.array([it[0] for it in content])
                self.v_duration = np.array([it[1] for it in content])
                self.v_contrast = np.array([it[2] for it in content])
                self.v_ori = np.array([it[3] for it in content])
                self.v_phase = np.array([it[4] for it in content])
                self.v_pos = np.array([it[5] for it in content])
                self.v_size = np.array([it[6] for it in content])
                self.v_flick = np.array([it[7] for it in content])
                self.v_interpolate = np.array([it[8] for it in content], dtype=bool)
                self.v_mask = np.array([it[9] is not None for it in content], dtype=bool)
                self.v_sf = np.array([it[10] for it in content])
                self.v_tf = np.array([it[11] for it in content])
                self.v_opto = np.array([it[12] for it in content], dtype=int)
                self.v_pattern = np.array([it[13] for it in content])

                remove_code.append(code)

            elif self.log_info[code] == 'PhotoIndicator':
                self.p_time = np.array([it[0] for it in content])
                self.p_state = np.array([it[1] for it in content], dtype=bool)
                self.p_size = np.array([it[2] for it in content])
                self.p_pos = np.array([it[3] for it in content])
                self.p_units = np.array([it[4] for it in content])
                self.p_mode = np.array([it[5] for it in content], dtype=int)
                self.p_frames = np.array([it[6] for it in content], dtype=int)
                self.p_enabled = np.array([it[7] for it in content], dtype=bool)

                remove_code.append(code)

        for code in remove_code:
            del log_data[code]

        return log_data

    def get_visual_presentation_dataframe(self) -> pl.DataFrame:
        """
        ┌────────────┬──────────┬──────────┬─────┬───┬──────┬─────┬──────┬─────────┐
        │ time       ┆ duration ┆ contrast ┆ ori ┆ … ┆ sf   ┆ tf  ┆ opto ┆ pattern │
        │ ---        ┆ ---      ┆ ---      ┆ --- ┆   ┆ ---  ┆ --- ┆ ---  ┆ ---     │
        │ f64        ┆ i64      ┆ i64      ┆ i64 ┆   ┆ f64  ┆ i64 ┆ i64  ┆ str     │
        ╞════════════╪══════════╪══════════╪═════╪═══╪══════╪═════╪══════╪═════════╡
        │ 18.990026  ┆ 10       ┆ 1        ┆ 0   ┆ … ┆ 0.04 ┆ 10  ┆ 0    ┆ square  │
        │ 21.000029  ┆ 10       ┆ 1        ┆ 0   ┆ … ┆ 0.04 ┆ 10  ┆ 0    ┆ square  │
        │ …          ┆ …        ┆ …        ┆ …   ┆ … ┆ …    ┆ …   ┆ …    ┆ …       │
        │ 619.054972 ┆ 10       ┆ 1        ┆ 0   ┆ … ┆ 0.04 ┆ 50  ┆ 0    ┆ square  │
        │ 619.084972 ┆ 10       ┆ 1        ┆ 0   ┆ … ┆ 0.04 ┆ 50  ┆ 0    ┆ square  │
        └────────────┴──────────┴──────────┴─────┴───┴──────┴─────┴──────┴─────────┘
        :return:
        """
        df = pl.DataFrame().with_columns(
            pl.Series(self.v_present_time).alias('time'),
            pl.Series(self.v_duration).alias('duration'),
            pl.Series(self.v_contrast).alias('contrast'),
            pl.Series(self.v_ori).alias('ori'),
            pl.Series(self.v_phase).alias('phase'),
            pl.Series(self.v_pos).alias('pos'),
            pl.Series(self.v_size).alias('size'),
            pl.Series(self.v_flick).alias('flick'),
            pl.Series(self.v_interpolate).alias('interpolate'),
            pl.Series(self.v_mask).alias('mask'),
            pl.Series(self.v_sf).alias('sf'),
            pl.Series(self.v_tf).alias('tf'),
            pl.Series(self.v_opto).alias('opto'),
            pl.Series(self.v_pattern).alias('pattern')
        )
        return df

    def get_photo_indicator_dataframe(self) -> pl.DataFrame:
        """
        ┌────────────┬───────┬──────┬────────────┬───────┬──────┬────────┬────────┐
        │ time       ┆ state ┆ size ┆ pos        ┆ units ┆ mode ┆ frames ┆ enable │
        │ ---        ┆ ---   ┆ ---  ┆ ---        ┆ ---   ┆ ---  ┆ ---    ┆ ---    │
        │ f64        ┆ bool  ┆ i64  ┆ list[i64]  ┆ str   ┆ i64  ┆ i64    ┆ bool   │
        ╞════════════╪═══════╪══════╪════════════╪═══════╪══════╪════════╪════════╡
        │ 18.990026  ┆ false ┆ 60   ┆ [740, 370] ┆ pix   ┆ 0    ┆ 20     ┆ true   │
        │ 21.000029  ┆ true  ┆ 60   ┆ [740, 370] ┆ pix   ┆ 0    ┆ 20     ┆ true   │
        │ …          ┆ …     ┆ …    ┆ …          ┆ …     ┆ …    ┆ …      ┆ …      │
        │ 607.094955 ┆ false ┆ 60   ┆ [740, 370] ┆ pix   ┆ 0    ┆ 20     ┆ true   │
        │ 609.104958 ┆ true  ┆ 60   ┆ [740, 370] ┆ pix   ┆ 0    ┆ 20     ┆ true   │
        └────────────┴───────┴──────┴────────────┴───────┴──────┴────────┴────────┘
        :return:
        """
        df = pl.DataFrame().with_columns(
            pl.Series(self.p_time).alias('time'),
            pl.Series(self.p_state).alias('state'),
            pl.Series(self.p_size).alias('size'),
            pl.Series(self.p_pos).alias('pos'),
            pl.Series(self.p_units).alias('units'),
            pl.Series(self.p_mode).alias('mode'),
            pl.Series(self.p_frames).alias('frames'),
            pl.Series(self.p_enabled).alias('enable')
        )
        return df

    def get_state_machine_dataframe(self) -> pl.DataFrame:
        """
        ┌────────────┬───────────────────────────┬───────────────────────────┐
        │ time       ┆ state                     ┆ prev_state                │
        │ ---        ┆ ---                       ┆ ---                       │
        │ f64        ┆ object                    ┆ object                    │
        ╞════════════╪═══════════════════════════╪═══════════════════════════╡
        │ 18.990026  ┆ ('States.SHOW_BLANK', 1)  ┆ ('States.STIM_SELECT', 0) │
        │ 20.990029  ┆ ('States.SHOW_STIM', 2)   ┆ ('States.SHOW_BLANK', 1)  │
        │ …          ┆ …                         ┆ …                         │
        │ 609.094958 ┆ ('States.SHOW_STIM', 2)   ┆ ('States.SHOW_BLANK', 1)  │
        │ 619.094972 ┆ ('States.STIM_SELECT', 0) ┆ ('States.SHOW_STIM', 2)   │
        └────────────┴───────────────────────────┴───────────────────────────┘
        :return:
        """
        df = pl.DataFrame().with_columns(
            pl.Series(it[0] for it in self.log_data[2]).alias('time'),
            pl.Series(it[1] for it in self.log_data[2]).alias('state'),
            pl.Series(it[2] for it in self.log_data[2]).alias('prev_state'),
        )
        return df

    def get_log_dict_dataframe(self) -> pl.DataFrame:
        """
        ┌────────────┬──────────┬──────────┬──────────────┬────────────┐
        │ time       ┆ block_nr ┆ trial_nr ┆ condition_nr ┆ trial_type │
        │ ---        ┆ ---      ┆ ---      ┆ ---          ┆ ---        │
        │ f64        ┆ i64      ┆ i64      ┆ i64          ┆ i64        │
        ╞════════════╪══════════╪══════════╪══════════════╪════════════╡
        │ 18.990026  ┆ 0        ┆ 0        ┆ 0            ┆ 1          │
        │ 30.990043  ┆ 1        ┆ 0        ┆ 0            ┆ 1          │
        │ …          ┆ …        ┆ …        ┆ …            ┆ …          │
        │ 595.083939 ┆ 48       ┆ 0        ┆ 0            ┆ 1          │
        │ 607.094955 ┆ 49       ┆ 0        ┆ 0            ┆ 1          │
        └────────────┴──────────┴──────────┴──────────────┴────────────┘
        :return:
        """
        df = pl.DataFrame().with_columns(
            pl.Series(it[0] for it in self.log_data[3]).alias('time'),
            pl.Series(it[1] for it in self.log_data[3]).alias('block_nr'),
            pl.Series(it[2] for it in self.log_data[3]).alias('trial_nr'),
            pl.Series(it[3] for it in self.log_data[3]).alias('condition_nr'),
            pl.Series(it[4] for it in self.log_data[3]).alias('trial_type'),
        )

        return df

    def get_column(self,
                   category: int | str,
                   field: str,
                   dtype=float,
                   mapper: Callable[[Any], Any] | None = None) -> np.ndarray:

        if category in ('Gratings', 'FunctionBased', 'PhotoIndicator'):
            raise RuntimeError(f'preprocessed log : {category}:{field}')

        if isinstance(category, str):
            category = self._get_category_code(category)

        field = self._get_field_index(category, field)
        data = [it[field] for it in self.log_data[category]]

        if mapper is not None:
            # i.e., operator.itemgetter(1) for tuple statemachine ('States.SHOW_STIM', 2)...
            data = list(map(mapper, data))

        return np.array(data, dtype=dtype)

    def _get_category_code(self, category: str) -> int:
        for code, name in self.log_info.items():
            if name == category:
                return code
        raise KeyError('')

    def _get_field_index(self, category: int, field: str) -> int:
        header = self.log_header[category]
        if field == 'time':
            return 0
        return header.index(field) + 1

    @property
    def exp_start_time(self) -> float:
        return self.riglog_data.dat[0, 2] / 1000

    @property
    def exp_end_time(self) -> float:
        return self.riglog_data.dat[-1, 2] / 1000

    @property
    def stim_start_time(self) -> float:
        return self.p_time[1] + self.time_offset

    @property
    def stim_end_time(self) -> float:
        return self.v_present_time[-1] + self.time_offset

    @property
    def stimulus_segment(self) -> np.ndarray:
        # ret = self.riglog_data.screen_event[:, 0].reshape(-1, 2)  # directly use riglog time
        # stimlog bug starting from `diode off`, and ending from `diode True`
        # thus remove first point and add last point manually
        _p_time = np.concatenate((self.p_time[1:], np.array([self.v_present_time[-1]])))
        ret = _p_time.reshape(-1, 2) + self.time_offset
        return ret

    def session_trials(self) -> dict[Session, SessionInfo]:
        return {
            prot.name: prot
            for prot in get_protocol_sessions(self)
        }

    @property
    def time_offset(self) -> float:
        if self._cache_time_offset is None:
            # noinspection PyTypeChecker
            self._cache_time_offset = _time_offset(self.riglog_data, self, self.diode_offset)[0]
        return self._cache_time_offset

    def get_stim_pattern(self) -> StimPattern:
        prot = StimpyProtocol.load(self.stimlog_file.with_suffix('.prot'))
        log_nr = self.get_column('LogDict', 'condition_nr').astype(int)

        t = self.stimulus_segment
        ori = np.array([prot['ori'][n] for n in log_nr])
        sf = np.array([prot['sf'][n] for n in log_nr])
        tf = np.array([prot['tf'][n] for n in log_nr])
        contrast = np.array([prot['c'][n] for n in log_nr])
        dur = np.array([prot['dur'][n] for n in log_nr])

        return StimPattern(t, ori, sf, tf, contrast, dur)

    def get_time_profile(self):
        raise NotImplementedError('')


# ========== #

@deprecated(reason='generalized, to sequential offset method')
def _time_offset(rig: RiglogData,
                 stm: StimlogGit,
                 diode_offset=True) -> tuple[float, float]:
    """
    time offset used in the `new stimpy`
    offset time from screen_time .riglog (diode) to .stimlog
    ** normally stimlog time value are larger than riglog

    :param rig:
    :param stm:
    :param diode_offset: whether correct diode signal
    :return: tuple of offset average and std value
    """
    if not isinstance(stm, StimlogGit):
        raise TypeError('')

    screen_time = rig.screen_event.time

    if diode_offset:
        try:
            # new stimpy might be a negative value (stimlog time value larger than riglog)
            # stimlog bug starting from `diode off`, and ending from `diode True`
            # thus remove first point and add last point manually
            _p_time = np.concatenate((stm.p_time[1:], np.array([stm.v_present_time[-1]])))
            offset_t = screen_time[::2] - _p_time[::2]

        except ValueError as e:
            fprint(f'number of diode pulse and stimulus mismatch from {e}', vtype='error')
            fprint(f'use the first pulse diff for alignment', vtype='error')
            offset_t = screen_time[0] - stm.p_time[1]
    else:
        raise NotImplementedError('')

    offset_t_avg = float(np.mean(offset_t))
    offset_t_std = float(np.std(offset_t))

    fprint(f'time offset between stimlog and riglog: {round(offset_t_avg, 3)}')
    fprint(f'offset_std: {round(offset_t_std, 3)}')

    return offset_t_avg, offset_t_std
