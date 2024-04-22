import contextlib
import functools
from collections.abc import Callable
from typing import Union, Literal, Iterator, TypeVar

import numpy as np

from neuralib.persistence import PersistenceHandler
from neuralib.persistence import persistence_class, field
from .base import ProcessedEphysRecording

__all__ = [
    'ProcessedEphysRecordingMeta',
    'save', 'load', 'open', 'clone',
    'per_channel',
    'global_car', 'local_car',
]


@persistence_class(name='ephys')
class ProcessedEphysRecordingMeta:
    filename: str = field(validator=True, filename=True)
    process: str = field(validator=True, filename=True)

    channel_list: np.ndarray  # (C,)
    """channel list of this signals, or its source channels"""

    time_start: float  # sec
    """start time of signals"""

    total_samples: int  # (S,)
    """total signal samples per channels"""

    sample_rate: float  # 1/sec
    """sample rate"""

    dtype: np.dtype

    meta: dict[str, str]

    @property
    def total_channels(self) -> int:
        return len(self.channel_list)

    @property
    def time(self) -> np.ndarray:
        return np.arange(self.total_samples) / self.sample_rate + self.time_start


Handler = TypeVar('Handler', bound=PersistenceHandler[ProcessedEphysRecordingMeta])


def save(handler: Handler, meta: ProcessedEphysRecordingMeta, data: ProcessedEphysRecording):
    meta.channel_list = data.channel_list
    meta.time_start = data.time_start
    meta.total_samples = data.total_samples
    meta.sample_rate = data.sample_rate
    meta.dtype = data.dtype
    meta.meta = dict(data.meta)

    meta_path = handler.filepath(meta)
    handler.save_persistence(meta, meta_path)

    with open(handler, meta) as mmap:
        mmap[:] = data[:]


def load(handler: Handler, meta: ProcessedEphysRecordingMeta) -> ProcessedEphysRecording:
    meta_path = handler.filepath(meta)
    if not (data_path := meta_path.with_suffix('.bin')).exists():
        raise FileNotFoundError(data_path)

    shape = (meta.total_channels, meta.total_samples)
    mmap = np.memmap(data_path, shape=shape, dtype=meta.dtype, mode='r', offset=0, order='F')

    return ProcessedEphysRecording(meta.time, meta.channel_list, mmap, meta=meta.meta)


@contextlib.contextmanager
def open(handler: Handler, meta: ProcessedEphysRecordingMeta) -> Iterator[np.memmap]:
    meta_path = handler.filepath(meta)

    data_path = meta_path.with_suffix('.bin')
    temp_path = data_path.with_suffix('.tmp')
    shape = (meta.total_channels, meta.total_samples)
    mmap = np.memmap(temp_path, shape=shape, dtype=meta.dtype, mode='w+', offset=0, order='F')
    try:
        yield mmap
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    else:
        temp_path.rename(data_path)


def clone(dst: Union[str, ProcessedEphysRecordingMeta], src: ProcessedEphysRecordingMeta, *,
          channel_list: np.ndarray = None,
          time_start: float = None,
          total_samples: int = None,
          sample_rate: float = None,
          dtype: np.dtype = None,
          meta: dict[str, str] = None) -> ProcessedEphysRecordingMeta:
    if isinstance(dst, str):
        dst = ProcessedEphysRecordingMeta(src.filename, dst)

    dst.channel_list = channel_list if channel_list is not None else src.channel_list.copy()
    dst.time_start = time_start if time_start is not None else src.time_start
    dst.total_samples = total_samples if total_samples is not None else src.total_samples
    dst.sample_rate = sample_rate if sample_rate is not None else src.sample_rate
    dst.dtype = dtype if dtype is not None else src.dtype
    dst.meta = meta if meta is not None else dict(src.meta)
    return dst


def per_channel(handler: Handler,
                process: str,
                func: Callable[[int, np.ndarray], np.ndarray],
                meta: ProcessedEphysRecordingMeta, *,
                chunk: int = 4) -> ProcessedEphysRecordingMeta:
    ret = clone(process, meta)

    signals = load(handler, meta)
    with open(handler, ret) as ret_data:
        for i in range(0, meta.total_channels, chunk):
            j = min(i + chunk, meta.total_samples)

            cache = signals[i:j]
            for k in range(j - i):
                c = int(meta.channel_list[i + k])
                ret_data[i + k] = func(c, cache[k])

    return ret


def global_car(handler: Handler, meta: ProcessedEphysRecordingMeta, *,
               method: Literal['mean', 'median'] = 'median',
               chunk: int = 128) -> ProcessedEphysRecordingMeta:
    if method == 'mean':
        method = np.mean
    elif method == 'median':
        method = np.median
    else:
        raise ValueError()

    ret = clone('compute_command_artifact', meta, channel_list=np.array([-1]))

    signals = load(handler, meta)
    with open(handler, ret) as ret_data:
        for i in range(0, meta.total_samples, chunk):
            sample_slice = slice(i, min(i + chunk, meta.total_samples))
            ret_data[sample_slice] = method(signals[:, sample_slice], axis=0)

    return ret


def local_car(handler: Handler, meta: ProcessedEphysRecordingMeta, *,
              method: Literal['mean', 'median'] = 'median',
              radius: Union[float, tuple[float, float]] = (30, 100),
              chunk: int = 128) -> ProcessedEphysRecordingMeta:
    oper = build_local_car_operator(None, radius)  # TODO

    if method == 'mean':
        def method(data):
            return oper @ data
    elif method == 'median':
        method = functools.partial(apply_local_car_median, oper)
    else:
        raise ValueError()

    ret = clone('local_car', meta, channel_list=np.array([-1]))
    signals = load(handler, meta)

    with open(handler, ret) as ret_data:
        for i in range(0, meta.total_samples, chunk):
            sample_slice = slice(i, min(i + chunk, meta.total_samples))
            ret_data[sample_slice] = method(signals[:, sample_slice])

    return ret


def build_local_car_operator(pos: np.ndarray, radius: Union[float, tuple[float, float]]) -> np.ndarray:
    """

    :param pos: (C, 2) channel position array
    :param radius:
    :return: (C, C) operator
    """
    x = pos[:, 0]
    y = pos[:, 1]
    dx = np.abs(np.subtract.outer(x, x))
    dy = np.abs(np.subtract.outer(y, y))
    dd = np.sqrt(dx ** 2 + dy ** 2)

    n = len(pos)
    ret = np.zeros_like(dd)

    if isinstance(radius, (int, float)):
        for i in range(n):
            c = dd[i] <= radius
            if np.any(c):
                ret[i, c] = -1 / np.count_nonzero(c)
            ret[i, i] = 1
    elif isinstance(radius, tuple):
        exclude, radius = radius
        for i in range(n):
            c = (exclude <= dd[i]) & (dd[i] <= radius)
            if np.any(c):
                ret[i, c] = -1 / np.count_nonzero(c)
            ret[i, i] = 1

    else:
        raise TypeError()

    return ret


def apply_local_car_median(oper: np.ndarray, data: np.ndarray) -> np.ndarray:
    """

    :param oper: (C, C) operator
    :param data: (C, S) array
    :return: (C, S)
    """
    c, _ = data.shape
    a, b = np.nonzero(oper < 0)
    med = np.zeros_like(data, dtype=float)
    for i in range(c):
        if len(j := b[a == i]):
            med[i] = np.median(data[j], axis=0)
    return data - med
