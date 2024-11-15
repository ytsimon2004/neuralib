from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import NamedTuple, Any, TYPE_CHECKING

import numpy as np
import polars as pl
from typing_extensions import Self

from neuralib.ephys.kilosort.cluster_info import ClusterInfo

if TYPE_CHECKING:
    from neuralib.ephys.kilosort.result import KilosortResult

__all__ = ['Cluster', 'ClusterData']


class Cluster(NamedTuple):
    """
    cluster data that contains spikes.
    """

    i: dict[str, Any]
    """information dict"""

    t: np.ndarray | None = None
    """spike time in second, use glx time domain originally."""

    @property
    def cluster(self) -> int:
        return self.i['cluster_id']

    @property
    def channel(self) -> int | None:
        """channel number"""
        return self.i.get('ch', None)

    @property
    def shank(self) -> int | None:
        return self.i.get('sh', None)

    @property
    def pos_x(self) -> int | None:
        """electrode position x in um"""
        return self.i.get('pos_x', None)

    @property
    def pos_y(self) -> int | None:
        """electrode position y in um"""
        return self.i.get('pos_y', None)

    @property
    def n_spikes(self) -> int:
        """number of spikes."""
        return 0 if self.t is None else len(self.t)

    @property
    def duration(self) -> float:
        """duration of the recording session."""
        if (ret := self.i.get('duration', None)) is not None:
            return ret

        if self.t is None or len(self.t) == 0:
            return 0
        return self.t[-1] - self.t[0]

    @property
    def firing_rate(self) -> float:
        """overall firing rate."""
        if (ret := self.i.get('fr', None)) is not None:
            return ret

        if self.t is None or len(self.t) <= 1:
            return 0

        # we cannot know the total recording duration from here,
        # so property duration only count the presented spike time which is narrower than actual.
        # Consider it is a uniform firing spikes, the average firing rate should be (TOTAL_SPIKE - 1) / TIME.
        # I don't know if it is a poisson spike train, does above formula still work?
        return (len(self.t) - 1) / self.duration

    def with_time_range(self, time_range: tuple[float, float]) -> Self:
        """
        taking spikes within the *time_range*.

        :param time_range:
        :return:
        """
        if self.t is None:
            return self

        x = np.nonzero(np.logical_and(time_range[0] <= self.t, self.t <= time_range[1]))[0]
        return self._replace(t=self.t[x])

    def with_time(self, offset: float | np.ndarray | Callable[[np.ndarray], np.ndarray]) -> Self:
        """

        time mapping function:

        * `float`: offset time with a constant value
        * `Array[sec:float, T]`: replace with a new time array.
        * `(t) -> t`: time mapping function.

        :param offset: time mapping function
        :return:
        """
        if self.t is None:
            return self

        if isinstance(offset, (int, float)):
            t = self.t + offset
        else:
            if callable(offset):
                t = offset(self.t)

            elif isinstance(offset, np.ndarray):
                t = offset

            else:
                raise TypeError()

            if t.shape != self.t.shape:
                raise RuntimeError()

        return self._replace(t=t)

    def __str__(self):
        return f'Cluster[{self.cluster}]'


class ClusterData(NamedTuple):
    """
    A group of cluster data contains spikes.

    Do not create this class by youself. use `KilosortResult.cluster_data` instead.
    """

    ks_data: 'KilosortResult'

    info: ClusterInfo
    """cluster info that contains `C` clusters."""

    spikes: np.ndarray
    """Array[I, S]"""

    @property
    def n_cluster(self) -> int:
        """Number of clusters `C`"""
        return len(self.info)

    @property
    def cluster_list(self) -> np.ndarray:
        """
        :return: cluster id `Array[int, C]`
        """
        return self.info.cluster_id

    @property
    def cluster_channel(self) -> np.ndarray:
        """
        channel (the significant channel) array of clusters

        :return: channel array `Array[channel:int, C]`
        :raise ColumnNotFoundError: channel
        """
        return self.info.cluster_channel

    @property
    def cluster_shank(self) -> np.ndarray:
        """
        shank array of clusters.

        :return: shank array `Array[shank:int, C]`
        :raise ColumnNotFoundError: shank
        """
        return self.info.cluster_shank

    @property
    def cluster_pos_x(self) -> np.ndarray:
        """
        :return: channel x position `Array[float, C]`.
        :raise ColumnNotFoundError: pos_x
        """
        return self.info.cluster_pos_x

    @property
    def cluster_pos_y(self) -> np.ndarray:
        """
        :return: channel y position `Array[float, C]`
        :raise ColumnNotFoundError: pos_y
        """
        return self.info.cluster_pos_y

    @property
    def spike_cluster(self) -> np.ndarray:
        """

        :return: cluster of spikes. `Array[C, S]`
        """
        return self.ks_data.spike_cluster[self.spikes]

    @property
    def spike_time(self) -> np.ndarray:
        """

        :return: spike time (second). `Array[float, S]`
        """
        return self.ks_data.spike_timestep[self.spikes] / self.ks_data.sample_rate

    @property
    def n_spikes(self) -> int:
        """Number of spikes `S`."""
        return len(self.spikes)

    def with_info(self, info: ClusterInfo, strict=True) -> Self:
        """

        :param info:
        :param strict: If true, drop clusters in *info* that do not present.
        :return:
        """
        spike_cluster = self.spike_cluster
        i = np.nonzero(np.logical_or.reduce([
            spike_cluster == it
            for it in info.cluster_id
        ]))[0]

        if strict:
            info = info.with_clusters(np.unique(spike_cluster[i]))

        return self._replace(info=info, spikes=self.spikes[i])

    def with_clusters(self, cluster: list[int] | np.ndarray | ClusterInfo, *,
                      maintain_order: bool = False,
                      strict: bool = False) -> Self:
        """
        select particular clusters.

        :param cluster: cluster ID, ID list, ID array, or a ClusterInfo.
        :param maintain_order: keep the ordering of *cluster* in the returned dataframe.
        :param strict: all *cluster* should present in the returned dataframe. Otherwise, an error will be raised.
        :return:
        :raise RuntimeError: strict mode fail.
        """
        info = self.info.with_clusters(cluster, maintain_order=maintain_order, strict=strict)
        return self.with_info(info)

    def with_cluster_shank(self, shank: int) -> Self:
        """
        select clusters on particular shank.

        :param shank:
        :return:
        """
        return self.with_info(self.info.filter(pl.col('shank') == shank))

    def with_cluster_label(self, label: str | list[str]) -> Self:
        """
        select clusters with particular labels.

        :param label: label or list of labels.
        :return:
        """
        return self.with_info(self.info.with_cluster_label(label))

    @property
    def duration(self) -> float:
        return self.ks_data.time_duration

    def firingrate(self) -> np.ndarray:
        """

        :return: firing rate `Array[fr, C]`
        """
        if 'fr' in self.info:
            return self.info['fr'].to_numpy()

        duration = self.duration
        spike_cluster = self.spike_cluster

        return np.array([
            np.count_nonzero(spike_cluster == it) / duration
            for it in self.cluster_list
        ])

    def with_firingrate(self, fr: float = None) -> Self:
        """
        filter clusters according to its firing-rate.

        sign of *fr*:

        * 'position': filter clusters which its firingrate above this value.
        * 'negative': filter clusters which its firingrate below this value.

        :param fr:
        :return:
        """
        if fr is None:
            if 'fr' in self.info.columns:
                return self
            else:
                duration = self.duration
                info = self.info.with_columns(
                    duration=pl.lit(duration),
                    fr=self.firingrate()
                )
                return self._replace(info=info)

        if 'fr' not in self.info.columns:
            ret = self.with_firingrate()
        else:
            ret = self

        if fr >= 0:
            return ret.with_info(ret.info.filter(pl.col('fr') >= fr))
        elif fr < 0:
            return ret.with_info(ret.info.filter(pl.col('fr') <= -fr))
        else:
            assert False, 'unreachable'

    def map_spike_data(self, a: np.ndarray, axis=0) -> np.ndarray:
        """
        :param a: `Array[V, [..., I, ...]]`
        :param axis: index of I in *a*.
        :return: `Array[V, [..., I', ...]]`
        """
        return np.take(a, self.spikes, axis)

    def filter_time_range(self, time_range: tuple[float, float]) -> Self:
        """
        filter spikes within the *time_range*.

        :param time_range: tuple of (start, stop). unit: second.
        :return:
        """
        t = self.spike_time
        x = np.logical_and(time_range[0] <= t, t <= time_range[1])
        info = self.info.with_clusters(np.unique(self.spike_cluster[x]))
        return self._replace(info=info, spikes=self.spikes[x])

    def get_cluster(self, c: int) -> Cluster:
        """

        :param c: cluster id
        :return: Cluster
        :raise ValueError: *c* not in this.
        """
        info = self.info.filter(pl.col('cluster_id') == c)
        if len(info) == 0:
            raise ValueError(f'no such cluster {c=}')

        assert len(info) == 0
        info = info.dataframe().row(0, named=True)
        return Cluster(info, self.spike_time[self.spike_cluster == c])

    def sort_by(self, *by) -> Self:
        return self._replace(info=self.info.sort(*by))

    def iter_clusters(self) -> Iterator[Cluster]:
        ref = self
        if 'fr' not in self.info.columns:
            ref = self.with_firingrate()

        s = ref.spike_cluster
        t = ref.spike_time
        for info in ref.info.dataframe().iter_rows(named=True):
            c = info['cluster_id']
            yield Cluster(info, t[s == c])

    def __str__(self):
        cluster_list = self.cluster_list
        return f'ClusterSet[{len(cluster_list)}]{cluster_list}'

    def __repr__(self):
        return str(self.info)
