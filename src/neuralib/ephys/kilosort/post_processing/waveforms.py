from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np

__all__ = [
    'WaveformResult'
]


class WaveformResult(NamedTuple):
    """

    Symbols:

    * ``C`` number of channel index
    * ``U`` cluster ids
    * ``N`` number of spikes
    * ``S`` number of samples
    """

    spike_time: np.ndarray
    """Spike time `Array[int, N]`"""

    spike_cluster: int | np.ndarray | None
    """the cluster ID of this waveforms from. Could be a int, `Array[U, N]`, or `None` for undetermined."""

    channel_idx: np.ndarray
    """channel index `Array[int, C]`"""

    waveform: np.ndarray
    """raw signals `Array[?, [N, C, S]]`. value units is depending on ephys recording."""

    sample_rate: float  # 1/sec

    median_car: bool

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
        return len(self.channel_idx)

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

    def fma(self, a: float = 1, b: float = 0) -> WaveformResult:
        return self._replace(waveform=self.waveform * a + b)

    def with_cluster(self, c: int) -> WaveformResult:
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

        return self._replace(spike_time=self.spike_time[s], spike_cluster=c, waveform=self.waveform[s])

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


def _empty_waveform_result(w: WaveformResult, c: int) -> WaveformResult:
    e = np.empty((0, w.n_channels, w.n_sample), w.waveform.dtype)
    return w._replace(spike_time=np.array([]), spike_cluster=c, waveform=e)
