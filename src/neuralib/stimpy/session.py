"""
User-specific experimental session
====================================

- Aim to know the time information about of a specific session in a given protocol


Pipeline
---------------

- Use protocol name to infer which type of protocol ``ProtocolAlias``
- Base on riglog/stimlog information to create a dictionary with ``Session``: ``SessionInfo`` pairs for a specific protocol
- Use the methods in ``SessionInfo`` to do the masking/slicing... with the ``RigEvent``

"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .baselog import StimlogBase

import numpy as np
from typing_extensions import TypeAlias

__all__ = [
    'ProtocolAlias',
    'get_protocol_name',
    'get_protocol_sessions',
    #
    'Session',
    'SessionInfo'
]

ProtocolAlias: TypeAlias = str
Session: TypeAlias = str


def get_protocol_name(filename: Path | 'StimlogBase') -> ProtocolAlias:
    from .baselog import StimlogBase
    if isinstance(filename, StimlogBase):
        filename = filename.stimlog_file

    filename = filename.stem
    if '_ori_sqr_12dir_2tf_3sf_bas' in filename or 'ori' in filename:
        return 'visual_open_loop'
    elif '_LDL' in filename:
        return 'light_dark_light'
    elif '_black' in filename:
        return 'grey'
    else:
        raise RuntimeError(f'unknown fname >> {filename}')


def get_protocol_sessions(stim: 'StimlogBase') -> list[SessionInfo]:
    protocol = get_protocol_name(stim)
    if protocol == 'visual_open_loop':
        return _get_protocol_sessions_vol(stim)
    elif protocol == 'light_dark_light':
        return _get_protocol_sessions_ldl(stim)
    elif protocol == 'grey':
        return _get_protocol_sessions_grey(stim)
    else:
        raise RuntimeError(f'unknown protocol >> {protocol}')


def _get_protocol_sessions_vol(stim: 'StimlogBase') -> list[SessionInfo]:
    """get session info for visual open loop protocol"""
    t0 = stim.riglog_data.exp_start_time
    t1 = stim.stim_start_time  # diode synced
    t2 = stim.stim_end_time  # diode synced
    t3 = stim.riglog_data.exp_end_time
    return [
        SessionInfo('light', (t0, t1)),
        SessionInfo('visual', (t1, t2)),
        SessionInfo('dark', (t2, t3)),
        SessionInfo('all', (t0, t3))
    ]


def _get_protocol_sessions_ldl(stim: 'StimlogBase') -> list[SessionInfo]:
    # diode signal is no longer reliable, use .prot file value instead
    from .stimpy_core import StimpyProtocol
    prot = StimpyProtocol.load(stim.stimlog_file.with_suffix('.prot'))

    t0 = stim.riglog_data.exp_start_time
    t1 = prot.start_blank_duration
    t2 = prot.total_duration - prot.end_blank_duration
    t3 = stim.riglog_data.exp_end_time

    # print('test t', t0, t1, t2, t3)  # todo total time exceed around 4s?
    return [
        SessionInfo('light_bas', (t0, t1)),
        SessionInfo('dark', (t1, t2)),
        SessionInfo('light_end', (t2, t3)),
        SessionInfo('all', (t0, t3)),
    ]


def _get_protocol_sessions_grey(stim: 'StimlogBase') -> list[SessionInfo]:
    t0 = stim.riglog_data.exp_start_time
    t3 = stim.riglog_data.exp_end_time
    return [SessionInfo('all', (t0, t3))]


# ============ #

class SessionInfo(NamedTuple):
    """session name and the corresponding time start-end"""

    name: Session
    """name of this session"""

    time: tuple[float, float]
    """time start/end of this session"""

    def time_mask_of(self, t: np.ndarray) -> np.ndarray:
        """
        create a mask for time array *t*.

        :param t: 1d time array
        :return: mask for this session
        """
        return np.logical_and(self.time[0] < t, t < self.time[1])

    def in_range(self, time: np.ndarray,
                 value: np.ndarray = None,
                 error=True) -> tuple[Any, Any]:
        """
        Get the range (the first and last value) of value array in this session.

        :param time: 1d time array (T,)
        :param value: 1d value array. Shape should as same as *time* (T,)
        :param error: raise an error when empty.
        :return: tuple of first and last `value` or `time`.
        """
        x = self.time_mask_of(time)
        if value is not None:
            t = value[x]
        else:
            t = time[x]

        if len(t) == 0:
            if error:
                raise ValueError('empty in extracting value or time from time mask')
            return np.nan, np.nan

        return t[0], t[-1]

    def in_slice(self, time: np.ndarray,
                 value: np.ndarray,
                 error=True) -> slice:
        """
        Get the slice of value in this session

        :param time: 1d time array (T,)
        :param value: 1d value array. Shape should as same as *time* (T,)
        :param error: raise an error when empty.
        :return: slice of `value`
        """
        if not np.issubdtype(value.dtype, np.integer):
            raise ValueError()

        v = value[self.time_mask_of(time)]

        if len(v) == 0:
            if error:
                raise ValueError('empty in extracting value or time from time mask')
            return slice(0, 0)

        return slice(int(v[0]), int(v[-1]))
