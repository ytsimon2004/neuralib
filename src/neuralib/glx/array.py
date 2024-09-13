import os
import re
from pathlib import Path
from typing import Optional, Union, Callable

import numpy as np
from typing_extensions import Self

from .base import EphysRecording

__all__ = ['EphysArray']


class EphysArray(EphysRecording):
    def __init__(self, time: np.ndarray,
                 channels: np.ndarray,
                 data: np.ndarray,
                 meta: dict[str, str] = None):
        if time.ndim != 1 or channels.ndim != 1 or data.shape != (len(channels), len(time)):
            raise ValueError(f'{time.ndim=}, {channels.ndim=}, {data.shape=}')

        self.__t = time.astype(float)
        self.__c = channels.astype(int)
        self.__d = data
        self.__m = {}
        if meta is not None:
            self.__m.update(meta)

    @property
    def data_path(self) -> Optional[Path]:
        return None

    @property
    def channel_list(self) -> np.ndarray:
        return self.__c

    @property
    def total_samples(self) -> int:
        return len(self.__t)

    @property
    def total_duration(self) -> float:
        return float(self.__t[-1] - self.__t[0])

    @property
    def sample_rate(self) -> float:
        return 1 / np.median(np.diff(self.__t))

    @property
    def time_start(self) -> float:
        return float(self.__t[0])

    @property
    def t(self) -> np.ndarray:
        return self.__t

    @property
    def dtype(self) -> np.dtype:
        return self.__d.dtype

    def __getitem__(self, item):
        return self.__d[item]

    def __setitem__(self, key, value):
        self.__d[key] = value

    @property
    def __array_interface__(self):
        return self.__d.__array_interface__

    def __array__(self):
        return self.__d

    @property
    def meta(self) -> dict[str, str]:
        return self.__m

    """time axis"""

    def with_time_range(self, t: tuple[float, float]) -> Self:
        x = np.nonzero(np.logical_and(t[0] <= self.__t, self.__t <= t[1]))[0]

        if np.all(x):
            return self

        return EphysArray(self.__t[x], self.__c, allocating_barrier(self.__d, None, x))

    def with_time(self, t: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]) -> Self:
        if isinstance(t, (int, float, np.number)):
            t = self.__t + t
        elif isinstance(t, np.ndarray):
            if t.shape != self.__t.shape:
                raise ValueError()
        elif callable(t):
            t = t(self.__t)
        else:
            raise TypeError()

        return EphysArray(t, self.__c, self.__d)

    """channel axis"""

    def with_channels(self, channel: np.ndarray) -> Self:
        """

        :param channel: channel number 1d array
        :return:
        """
        return self.with_channel_mask(np.logical_or.reduce(np.equal.outer(self.channel_list, channel), axis=1))

    def with_channel_mask(self, channel: np.ndarray) -> Self:
        """

        :param channel: channel index 1d bool or index array
        :return:
        """
        return EphysArray(self.__t, self.__c[channel], allocating_barrier(self.__d, channel, None))

    """signal"""

    def fma(self, a: float, b: float = 0) -> Self:
        return EphysArray(self.__t, self.__c, allocating_barrier(self.__d) * a + b)

    def as_voltage(self) -> Self:
        from .spikeglx import GlxRecording
        # XXX Does other EphysRecording use different units?
        return EphysArray(self.__t, self.__c, allocating_barrier(self.__d, dtype=np.float64) * GlxRecording.VOLTAGE_FACTOR)

    def downsampling(self, factor: int) -> Self:
        from scipy.signal import decimate
        data = decimate(self.__d, factor, axis=1)
        _, s = data.shape
        t = np.linspace(self.t[0], self.t[-1], num=s)
        return EphysArray(t, self.__c, data)


ALLOCATION_LIMIT = None


def allocation_limit() -> int:
    global ALLOCATION_LIMIT
    if ALLOCATION_LIMIT is not None:
        return ALLOCATION_LIMIT

    limit = os.environ.get('NEURALIB_NUMPY_ALLOCATION_LIMIT', '2G')
    m = re.match(r'(?:(\d+)G)?(?:(\d+)M)?(?:(\d+)K)?(?:(\d+)B)?', limit)
    if m is None:
        return 2 ** 30

    g = m.group(1) or 0
    m = m.group(2) or 0
    k = m.group(3) or 0
    b = m.group(4) or 0
    ALLOCATION_LIMIT = int(g) * 2 ** 30 + int(m) * 2 ** 20 + int(k) * 2 ** 10 + int(b)
    return ALLOCATION_LIMIT


def allocating_barrier(data: np.ndarray,
                       channels: Union[int, slice, np.ndarray] = None,
                       samples: Union[int, slice, np.ndarray] = None,
                       dtype: np.dtype = None) -> np.ndarray:
    if data.ndim == 1:
        s = len(data)
        c = 1
    else:
        c, s = data.shape

    if isinstance(channels, int):
        c = channels
        channels = None
    elif isinstance(channels, slice):
        channels = np.arange(c)[channels]
        c = len(channels)
    elif isinstance(channels, np.ndarray):
        c = len(np.arange(c)[channels])
    elif channels is not None:
        raise TypeError(f'type(channels)={type(channels).__name__}')

    if isinstance(samples, int):
        s = samples
        samples = None
    elif isinstance(samples, slice):
        samples = np.arange(s)[samples]
        s = len(samples)
    elif isinstance(samples, np.ndarray):
        s = len(np.arange(s)[samples])
    elif samples is not None:
        raise TypeError(f'type(samples)={type(samples).__name__}')

    if dtype is None:
        size = c * s * data.dtype.itemsize  # B
    else:
        size = c * s * dtype.itemsize  # B

    size = size / 2 ** 30  # GB
    if size > allocation_limit():
        raise RuntimeError(f'{round(size, 1)}GB allocating')

    if channels is None and samples is None:
        ret = data
    elif data.ndim == 1 and samples is not None:
        ret = data[samples]
    else:
        samples = samples if samples is not None else slice(None)
        channels = channels if channels is not None else slice(None)
        ret = data[channels, samples]

    return ret.astype(dtype)
