from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional, Union, Callable

import numpy as np
from typing_extensions import Self

from .base import EphysRecording
from .channel_info import ChannelInfo

__all__ = ['EphysArray']


class EphysArray(EphysRecording):
    def __init__(self, time: np.ndarray,
                 channels: np.ndarray,
                 data: np.ndarray,
                 meta: dict[str, str] = None):
        """
        :param time: A numpy array representing time points. `Array[sec:float, T]`
        :param channels: A numpy array representing channel identifiers. `Array[int, C]`
        :param data: A 2-dimensional numpy array where rows correspond to channels and columns correspond to time points. `Array[D, [C, T]]`
        :param meta: An optional dictionary containing metadata as string key-value pairs. If provided, it will be used to update the internal metadata dictionary.
        """
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
        """channel list `Array[int, C]`"""
        return self.__c

    @property
    def total_samples(self) -> int:
        """total sample number `T`."""
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
        """`Array[sec:float, T]`"""
        return self.__t

    @property
    def dtype(self) -> np.dtype:
        """Data type `D`"""
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
        """

        :param t: time range (start, end). unit: second
        :return:
        """
        x = np.nonzero(np.logical_and(t[0] <= self.__t, self.__t <= t[1]))[0]

        if np.all(x):
            return self

        return EphysArray(self.__t[x], self.__c, allocating_barrier(self.__d, None, x))

    def with_time(self, t: Union[float, np.ndarray, Callable[[np.ndarray], np.ndarray]]) -> Self:
        """
        time mapping function:

        * `float`: offset time with a constant value
        * `Array[sec:float, T]`: replace with a new time array.
        * `(t) -> t`: time mapping function.

        :param t: time mapping function.
        :return:
        """
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

    def with_channels(self, channel: list[int] | np.ndarray | ChannelInfo) -> Self:
        """

        :param channel: channel number 1d array
        :return:
        """
        if isinstance(channel, ChannelInfo):
            channel = channel.channel

        channel = np.atleast_1d(channel)
        return self.with_channel_mask(np.logical_or.reduce(np.equal.outer(self.channel_list, channel), axis=1))

    def with_channel_mask(self, channel: list[int] | np.ndarray | ChannelInfo) -> Self:
        """

        :param channel: channel index 1d bool or index array
        :return:
        """
        if isinstance(channel, ChannelInfo):
            channel = channel.channel

        channel = np.atleast_1d(channel)
        return EphysArray(self.__t, self.__c[channel], allocating_barrier(self.__d, channel, None))

    """signal"""

    def fma(self, a: float, b: float = 0) -> Self:
        """
        scale data by `new_data = old_data * a + b`

        :param a: scale factor.
        :param b: offset factor. default is 0.
        :return:
        """
        return EphysArray(self.__t, self.__c, allocating_barrier(self.__d) * a + b)

    def as_voltage(self) -> Self:
        """
        scale data to change unit from raw value to micro-voltage.

        TODO Does other EphysRecording use different units?

        :return:
        """
        from .spikeglx import GlxRecording
        return EphysArray(self.__t, self.__c,
                          allocating_barrier(self.__d, dtype=np.float64) * GlxRecording.VOLTAGE_FACTOR)

    def downsampling(self, factor: int) -> Self:
        from scipy.signal import decimate
        data = decimate(self.__d, factor, axis=1)
        _, s = data.shape
        t = np.linspace(self.t[0], self.t[-1], num=s)
        return EphysArray(t, self.__c, data)


ALLOCATION_LIMIT = None


def allocation_limit() -> int:
    """
    Get current allocation limit number.

    configurations:

    * set module global variable `ALLOCATION_LIMIT`
    * set environment variable `NEURALIB_NUMPY_ALLOCATION_LIMIT`

    expression:

    `[?G][?M][?K][?B]`

    example:

    ```bash
    env NEURALIB_NUMPY_ALLOCATION_LIMIT=2G python ...
    ```

    :return: size in bytes.
    """
    global ALLOCATION_LIMIT
    if ALLOCATION_LIMIT is not None:
        return ALLOCATION_LIMIT

    limit = os.environ.get('NEURALIB_NUMPY_ALLOCATION_LIMIT', '2G')
    m = re.match(r'(?:(.+?)G)?(?:(,+?)M)?(?:(.+?)K)?(?:\d+B)?', limit)
    if m is None:
        return 2 ** 30

    g = m.group(1) or 0
    m = m.group(2) or 0
    k = m.group(3) or 0
    b = m.group(4) or 0
    ALLOCATION_LIMIT = float(g) * 2 ** 30 + float(m) * 2 ** 20 + float(k) * 2 ** 10 + int(b)
    return ALLOCATION_LIMIT


def allocating_barrier(data: np.ndarray,
                       channels: int | slice | np.ndarray = None,
                       samples: int | slice | np.ndarray = None,
                       dtype: np.dtype = None) -> np.ndarray:
    """
    Give a barrier before creating a big array, to prevent the process
    from out of memory and being crashing without any hint (program may
    get killed immediately by OS).

    :param data: source data `Array[?, [C, T]]`
    :param channels: slice on channel axis (the first axis).
    :param samples: slice on sample axis (the second axis).
    :param dtype: data type for new array
    :return: sliced data `Array[dtype, [C', T']]`
    :raise TypeError: incorrent *channels* or *samples* type
    :raise RuntimeError: new array over allocation limit.
    """
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
