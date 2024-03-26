from __future__ import annotations

from pathlib import Path

from neurolib.stimpy.baselog import StimlogBase
from neurolib.stimpy.session import SessionInfo
from neurolib.stimpy.stimpy_core import StimpyProtocol

__all__ = [
    'get_protocol_name',
    'get_protocol_sessions'
]


def get_protocol_name(filename: Path | StimlogBase) -> str:
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


def get_protocol_sessions(stim: StimlogBase) -> list[SessionInfo]:
    protocol = get_protocol_name(stim)
    if protocol == 'visual_open_loop':
        return _get_protocol_sessions_vol(stim)
    elif protocol == 'light_dark_light':
        return _get_protocol_sessions_ldl(stim)
    elif protocol == 'grey':
        return _get_protocol_sessions_grey(stim)
    else:
        raise RuntimeError(f'unknown protocol >> {protocol}')


def _get_protocol_sessions_vol(stim: StimlogBase) -> list[SessionInfo]:
    """
    calculate the time from `stimlog`. need to be use diode offset for correlating the time with `riglog`
    """
    t0 = stim.riglog_data.exp_start_time
    t1 = stim.stim_start_time
    t2 = stim.stim_end_time
    t3 = stim.riglog_data.exp_end_time
    return [
        SessionInfo('light', (t0, t1)),
        SessionInfo('visual', (t1, t2)),
        SessionInfo('dark', (t2, t3)),
        SessionInfo('all', (t0, t3))
    ]


def _get_protocol_sessions_ldl(stim: StimlogBase) -> list[SessionInfo]:
    filename = stim.stimlog_file
    if isinstance(filename, list):
        filename = filename[0]

    # diode signal is no longer reliable, use .prot file value instead
    prot = StimpyProtocol.load(filename.with_suffix('.prot'))
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


def _get_protocol_sessions_grey(stim: StimlogBase) -> list[SessionInfo]:
    t0 = stim.riglog_data.exp_start_time
    t3 = stim.riglog_data.exp_end_time
    return [SessionInfo('all', (t0, t3))]
