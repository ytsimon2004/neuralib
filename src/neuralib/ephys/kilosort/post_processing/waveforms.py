from __future__ import annotations

from collections.abc import Iterator, Mapping, Callable
from typing import NamedTuple

import numpy as np

from neuralib.ephys.kilosort.files import KilosortFiles
from neuralib.ephys.kilosort.result import KilosortResult

__all__ = [
    'WaveformResult',
    'get_waveforms'
]


class WaveformResult(NamedTuple):
    """

    Symbols:

    * ``C`` number of channel index
    * ``U`` cluster ids
    * ``N`` number of spikes
    * ``S`` number of samples
    * ``R`` ephys raw signal type
    """

    spike_time: np.ndarray
    """Spike time `Array[int, N]`"""

    spike_cluster: int | np.ndarray | None
    """the cluster ID of this waveforms from. Could be a int, `Array[U, N]`, or `None` for undetermined."""

    channel_list: int | np.ndarray
    """channel, or a array  `Array[int, C]`, or `Array[int, [N, C]]`"""

    waveform: np.ndarray
    """raw signals `Array[R, [N, C, S]]`."""

    sample_rate: float  # 1/sec

    median_car: bool

    @property
    def n_cluster(self) -> int | None:
        if self.spike_cluster is None:
            return None
        elif isinstance(self.spike_cluster, int):
            return 1
        else:
            return len(np.unique(self.spike_cluster))

    @property
    def cluster_list(self) -> np.ndarray | None:
        if self.spike_cluster is None:
            return None

        elif isinstance(self.spike_cluster, int):
            return np.array([self.spike_cluster], dtype=int)

        return np.unique(self.spike_cluster)

    @property
    def n_spikes(self) -> int:
        """N"""
        return len(self.spike_time)

    @property
    def n_channels(self) -> int:
        """C"""
        if isinstance(self.channel_list, int):
            return 1
        return self.channel_list.shape[-1]

    @property
    def n_sample(self) -> int:
        """S"""
        return self.waveform.shape[2]

    @property
    def duration(self) -> float:
        """second"""
        return self.n_sample / self.sample_rate

    @property
    def t(self) -> np.ndarray:
        """Array[sec, S]"""
        d = self.duration / 2
        s = self.n_sample
        return np.linspace(-d, d, s)

    def fma(self, a: float = 1, b: float = 0, t=float) -> WaveformResult:
        """
         `self * a + b`, and change `R` to `T`.

        :param a:
        :param b:
        :param t: cast type T
        :return: A WaveformResult with waveform shape `Array[T, [N, C, S]]`
        """
        return self._replace(waveform=(self.waveform * a + b).astype(t))

    def dot(self, a: np.ndarray = None, b: np.ndarray = None) -> WaveformResult:
        """
        `a @ self @ b`.

        **Note** If the `C` axis is not homo source, where `channel_list` is
        not an int value nor a 1-d array, think twice before applying a filter-like *a*.

        :param a: `Array[?, [C, C]]`
        :param b: `Array[?, [S, S]]`
        :return: A WaveformResult with waveform shape `Array[?, [N, C, S]]`
        """
        w = self.waveform
        if a is not None:
            w = np.vectorize(np.dot, signature='(c,c),(c,s)->(c,s)')(a, w)
        if b is not None:
            w = np.vectorize(np.dot, signature='(c,s),(s,s)->(c,s)')(w, b)
        return self._replace(waveform=w)

    def with_cluster(self, c: int) -> WaveformResult:
        """
        Take cluster *c*'s waveform.

        :param c:
        :return:
        """
        if self.spike_cluster is None:
            return self
        elif isinstance(self.spike_cluster, int):
            if self.spike_cluster == c:
                return self
            else:
                return _empty_waveform_result(self, c)

        s = np.nonzero(self.spike_cluster == c)[0]
        if len(s) == 0:
            return _empty_waveform_result(self, c)

        h = self.channel_list
        if isinstance(h, np.ndarray) and h.ndim == 2:
            h = h[s]

        return self._replace(spike_time=self.spike_time[s], spike_cluster=c, channel_list=h, waveform=self.waveform[s])

    def with_channel(self, c: int) -> WaveformResult:
        """
        Take waveform from given channel *c*.

        :param c:
        :return:
        """
        if isinstance(self.channel_list, int):
            if self.channel_list == c:
                return self
            else:
                return _empty_waveform_result(self, None, c)

        elif self.channel_list.ndim == 1:
            if np.count_nonzero(x := self.channel_list == c) == 1:
                return self._replace(channel_list=c, waveform=self.waveform[:, x, :])
            else:
                # It is nonsense we have duplicated channels,
                # so here should return an empty result.
                return _empty_waveform_result(self, None, c)

        elif self.channel_list.ndim == 2:
            ni, ci = np.nonzero(self.channel_list == c)
            if len(ni):
                # It is nonsense we have duplicated channels for every sample,
                # so ni should be a unique array
                assert len(ni) == len(np.unique(ni))
                t = self.spike_time[ni]

                s = self.spike_cluster[ni]
                if len(ss := np.unique(s)) == 1:
                    s = int(ss[0])

                return self._replace(spike_time=t, spike_cluster=s, channel_list=c, waveform=self.waveform[ni, ci, :])
            else:
                return _empty_waveform_result(self, None, c)
        else:
            assert False, 'unreachable'

    def __iter__(self) -> Iterator[WaveformResult]:
        if self.spike_cluster is None:
            return iter([self])
        elif isinstance(self.spike_cluster, int):
            return iter([self])
        else:
            return self.__iter_clusters()

    def __iter_clusters(self) -> Iterator[[WaveformResult]]:
        assert isinstance(self.spike_cluster, np.ndarray)
        for c in self.cluster_list:
            yield self.with_cluster(c)


def _empty_waveform_result(r: WaveformResult, c: int | None = None, h: int | np.ndarray = None) -> WaveformResult:
    if h is None:
        n_channels = r.n_channels

        h = r.channel_list
        if isinstance(h, np.ndarray):
            h = np.empty((0, n_channels), dtype=r.channel_list.dtype)
    elif isinstance(h, int):
        n_channels = 1
    elif h.ndim == 1:
        n_channels = len(h)
    else:
        n_channels = h.shape[-1]
        h = np.empty((0, n_channels), dtype=r.channel_list.dtype)

    w = np.empty((0, n_channels, r.n_sample), dtype=r.waveform.dtype)

    return r._replace(spike_time=np.array([]), spike_cluster=c, channel_list=h, waveform=w)


def get_waveforms(ks_data: KilosortFiles | KilosortResult,
                  cluster: int | np.ndarray,
                  channel: int | np.ndarray | Mapping[int, np.ndarray] = None, *,
                  duration: float = 3,
                  sample_times: int | Callable[[np.ndarray], np.ndarray] | None = None,
                  median_car: bool | Callable[[np.ndarray], np.ndarray] = False,
                  whitening=False) -> WaveformResult:
    """
    Extra spike waveform from kilosort result.

    **padding** If the waveform time window is at the beginning or at the end,
    the waveform will be padded with 0.

    :param ks_data:
    :param cluster: cluster number or spike index `Array[int, N]`
    :param channel: channel list `Array[int, C]`, or a dict `{U: Array[int, C]}`.
    :param duration: ms, correspond to number of sample `S`.
    :param sample_times: random pick certain number of spikes.
        It does nothing when it is `None`, or when the `N` is smaller than the *sample_times*.
        If could be a callable as a spike index (*cluster*) picking function
        with the signature `(Array[int, N]) -> Array[int, N*]`.
        It could be a replacement picking function.
        This function only sort the picking result without any extra checking.
    :param median_car: apply median car (common artifact removing) on all channels.
        It could be a callable with the signature `(Array[R, [C^, S]]) -> Array[R, S]`,
        where `C^` means all ephys channels. Make sure `R` type (ephys raw type) is kept in return.
    :param whitening: Ignored. TODO handle bad channels
    :return: a `WaveformResult`. If it has zero length on any axis (according to arguments),
        the returned result has always 0 on `N` axis (no matter it was zero channels or zero samples).
    :raises ValueError: illegal argument.
    :raises TypeError: illegal argument type or array shape.
    :raises KeyError: *cluster* not in the dict *channel*
    :raises FileNotFoundError: ephys recording missing.
    """
    if duration < 0:
        raise ValueError(f'negative duration : {duration}')

    if sample_times is not None and isinstance(sample_times, int) and sample_times < 0:
        raise ValueError(f'negative sample times : {sample_times}')

    if isinstance(ks_data, KilosortFiles):
        ks_data = ks_data.result()

    rec_file = ks_data.file.recording_data_file
    if not rec_file.exists():
        raise FileNotFoundError(str(rec_file))

    sample_rate = ks_data.sample_rate
    samples = int(duration * sample_rate / 1000)

    # cluster
    if isinstance(cluster, int) or np.isscalar(cluster):
        cluster = int(cluster)
        spike_index = np.nonzero(ks_data.spike_cluster == cluster)[0]
    elif isinstance(cluster, np.ndarray) and cluster.ndim == 1:
        spike_index = np.asarray(cluster, dtype=int)
        cluster = ks_data.spike_cluster[spike_index]
        if len(np.unique(cluster)) == 1:
            cluster = int(cluster[0])
    else:
        raise TypeError(f'cluster type : {cluster}')

    # sample cluster
    if sample_times is not None:
        if callable(sample_times):
            spike_index = np.sort(sample_times(spike_index))
        elif sample_times == 0:
            spike_index = np.array([], dtype=int)
        elif len(spike_index) > sample_times:
            i = np.arange(len(spike_index))
            np.random.shuffle(i)
            spike_index = np.sort(spike_index[i[:sample_times]])

    n_spikes = len(spike_index)

    # channel
    if isinstance(channel, int) or np.isscalar(channel):
        channel = int(channel)
        n_channel = 1
    elif isinstance(channel, np.ndarray) and channel.ndim == 1:
        n_channel = len(channel)
    elif isinstance(channel, dict):
        channels = np.unique([len(it) for it in channel.values()])
        if len(channels) != 1:
            raise ValueError('channel dict to not have same length')

        if isinstance(cluster, int):
            channel = np.asarray(channel[cluster])
            n_channel = len(channel)
            assert channel.ndim == 1
        else:
            n_channel = int(channels[0])
            channels = np.empty((n_spikes, n_channel), dtype=int)
            for i, t in enumerate(spike_index):
                channels[i] = channel[int(ks_data.spike_cluster[t])]
            channel = channels
    else:
        raise TypeError(f'channel type : {channel}')

    # empty case

    if n_spikes == 0 or n_channel == 0 or samples == 0:
        t = np.array([], dtype=float)
        w = np.empty((0, n_channel, samples), dtype=float)
        return WaveformResult(t, cluster, channel, w, sample_rate, median_car)

    rec_data = ks_data.file.ephys
    total_samples = rec_data.total_samples

    rec_value = rec_data[0, 0]
    waveform = np.empty((n_spikes, n_channel, samples), dtype=rec_value.dtype)

    for i, s in enumerate(spike_index):
        s = int(s)
        s1 = s - samples // 2
        s2 = s + samples // 2
        ss = slice(max(0, s1), min(s2, total_samples))

        if isinstance(channel, int):
            ch = [channel]
        elif channel.ndim == 1:
            ch = channel
        else:
            ch = channel[i]

        if median_car:
            chunk = rec_data[:, ss]

            if callable(median_car):
                median = median_car(chunk)
            else:
                # TODO how do I handle with sync channels?
                median = np.median(chunk, axis=0)

            if isinstance(rec_value, np.integer):
                median = median.astype(rec_value.dtype)

            chunk = rec_data[ch, ss] - median
        else:
            chunk = rec_data[ch, ss]

        _, size = chunk.shape
        if size == samples:
            waveform[i] = chunk
        elif s1 < 0:
            waveform[i] = 0
            waveform[i, :, samples - size:] = chunk
        else:
            assert total_samples < s2
            waveform[i] = 0
            waveform[i, :, :size] = chunk

    spike_time = ks_data.spike_time[spike_index]
    return WaveformResult(spike_time, cluster, channel, waveform, sample_rate, median_car)
