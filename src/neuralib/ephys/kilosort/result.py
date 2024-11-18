import functools
from typing import Optional, overload, Union

import numpy as np

from .cluster_data import ClusterData
from .cluster_info import ClusterInfo
from .files import KilosortFiles

__all__ = ['KilosortResult']


class KilosortResult:
    """

    Symbols:

    * ``C`` number of channel index
    * ``S`` number of spikes
    * ``U`` number of cluster/template ids

    Terminology:

    * `channel index`: an index array of channel number sequence which skipped the bad channel number.
    * `channel`, `channel number`, `channel list`: a value correspond to the Ephys recording channel.

    """

    def __init__(self, ks_file: KilosortFiles):
        self.file = ks_file
        self.para = ks_file.parameter_data()
        self.__tc_cache = {}

    @property
    def channel_number(self) -> int:
        """total channel number"""
        # TODO Is total of channel number or index?
        return self.para.channel_number

    @property
    def time_duration(self) -> Optional[float]:
        """total recording duration in seconds."""
        if (ephys := self.file.ephys) is None:
            return None

        return ephys.total_duration

    @property
    def sample_rate(self) -> float:
        return self.para.sample_rate

    @property
    def cluster_info(self) -> ClusterInfo:
        """
        read `cluster_info.tsv`.

        :return:
        :raises FileNotFoundError: `cluster_info.tsv` is not yet created (usually by Phy).
        """
        return self.file.cluster_info()

    @property
    def cluster_data(self) -> ClusterData:
        return ClusterData(self, self.cluster_info, np.arange(len(self.spike_cluster)))

    @functools.cached_property
    def channel_map(self) -> np.ndarray:
        """
        A mapping from channel index to channel number.

        :return: `Array[channel, C]`
        """
        info = self.file.ephys.channel_info()
        c_pos = np.column_stack([info.pos_x, info.pos_y])
        k_pos = self.channel_pos
        ret = np.zeros((len(k_pos),), dtype=int)
        for i in range(len(k_pos)):
            j = np.nonzero(np.logical_and(
                c_pos[:, 0] == k_pos[i, 0],
                c_pos[:, 1] == k_pos[i, 1],
            ))[0]
            if len(j) != 1:
                raise RuntimeError()
            ret[i] = j[0]
        return ret

    @overload
    def as_channel_list(self, channel_idx: int) -> int:
        pass

    @overload
    def as_channel_list(self, channel_idx: Union[list[int], np.ndarray]) -> np.ndarray:
        pass

    def as_channel_list(self, channel_idx):
        channel_idx = np.asarray(channel_idx)
        channel = self.channel_map[channel_idx]
        if channel_idx.ndim == 0:
            return int(channel)
        return channel

    @overload
    def as_channel_index(self, channel_list: int) -> int:
        pass

    @overload
    def as_channel_index(self, channel_list: Union[list[int], np.ndarray]) -> np.ndarray:
        pass

    def as_channel_index(self, channel_list):
        """

        >>> ks_data: KilosortResult
        >>> ks_data.channel_map
        [0, 1, 2, 3]
        >>> ks_data.as_channel_index(2)
        2
        >>> ks_data.as_channel_index(4)
        -1
        >>> ks_data.as_channel_index([2])
        [2]
        >>> ks_data.as_channel_index([2, 4])
        [2, -1]

        :param channel_list:
        :return:
        """
        channel_list = np.asarray(channel_list)
        i = np.searchsorted(self.channel_map, channel_list)
        if channel_list.ndim == 0:
            i = int(i)
            return i if self.channel_map[i] == channel_list else -1
        else:
            i[self.channel_map[i] != channel_list] = -1
            return i

    @functools.cached_property
    def channel_pos(self) -> np.ndarray:
        """

        :return: `Array[um, [C, 2]]`
        """
        # noinspection PyTypeChecker
        return np.load(self.file.channel_pos_file)

    def nearby_channels(self, channel_idx: int, *,
                        distance: Union[float, tuple[float, float]] = None,
                        count: int = None) -> np.ndarray:
        """

        :param channel_idx: channel index
        :param distance: um. channel position distance or channel (x, y) distance.
        :param count:
        :return: channel index array
        """
        pos = self.channel_pos
        c = pos[int(channel_idx)]
        dp = pos - c
        dx = np.abs(dp[:, 0])
        dy = np.abs(dp[:, 1])
        d = np.sqrt(dx ** 2 + dy ** 2)
        i = np.argsort(d)

        if distance is not None:
            if isinstance(distance, tuple):
                _x, _y = distance
                i = i[(dx[i] <= _x) & (dy[i] <= _y)]
            else:
                i = i[d[i] <= distance]

        if count is not None:
            i = i[:count]

        assert np.any(i == channel_idx)
        return i

    @functools.cached_property
    def spike_timestep(self) -> np.ndarray:
        """

        :return: Array[sample:int, S]
        """
        return np.load(self.file.spike_time_file).ravel()

    @functools.cached_property
    def spike_time(self) -> np.ndarray:
        """

        :return: `Array[second:float, S]`
        """
        return self.spike_timestep / self.sample_rate

    @functools.cached_property
    def spike_cluster(self) -> np.ndarray:
        """
        giving the cluster identity of each spike.

        This file is optional and if not provided will be automatically created the first time you run
        the template gui, taking the same values as spike_templates.npy until you do any merging or splitting.

        :return: `Array[U, S]`
        """
        return np.load(self.file.spike_cluster_file).ravel()

    @functools.cached_property
    def spike_amplitudes(self) -> np.ndarray:
        """
        the amplitude scaling factor that was applied to the template when extracting that spike.

        :return: `Array[amplitude:float, S]`
        """
        return np.load(self.file.spike_amplitude_file).ravel()

    @functools.cached_property
    def spike_template(self) -> np.ndarray:
        """
        specifying the identity of the template that was used to extract each spike

        :return: `Array[U, S]`
        """
        return np.load(self.file.spike_template_file).ravel()

    def template_data(self) -> np.ndarray:
        """Template data giving the template shapes on the channels given in templates_ind.npy.

        The first axis of template data match to the original cluster ID.
        If this data set has been manual correction by phy, that the cluster ID
        is not continuous anymore, so you cannot use new cluster ID to get to
        correspond template directly. Please use :meth:`get_template`

        :return: `Array[float, [U, S, C]]`
        """
        return np.load(self.file.template_file)

    @overload
    def get_template(self, cluster: int) -> int:
        pass

    @overload
    def get_template(self, cluster: Union[list[int], np.ndarray]) -> np.ndarray:
        pass

    def get_template(self, cluster):
        """Get the template index for certain cluster ID.

        :param cluster: cluster ID
        :return: template index. -1 if template not found.
        """
        cluster = np.asarray(cluster)  # (A,)
        if cluster.ndim == 0:
            template = self.spike_template[self.spike_cluster == cluster]
            if len(template) == 0:
                return -1
            template, count = np.unique(template, return_counts=True)
            return template[np.argmax(count)]
        else:
            return np.array([self.get_template(c) for c in cluster])

    @overload
    def get_channel(self, *, template: int) -> int:
        pass

    @overload
    def get_channel(self, *, cluster: int) -> int:
        pass

    @overload
    def get_channel(self, *, cluster: Union[list[int], np.ndarray]) -> np.ndarray:
        pass

    def get_channel(self, *, template=None, cluster=None):
        """
        :param template: template index
        :param cluster: cluster ID number or ID array
        :return: channel index
        """
        if template is None and cluster is None:
            raise RuntimeError('either template or cluster')
        if template is not None and cluster is not None:
            raise RuntimeError('either template or cluster')

        if cluster is not None:
            cluster = np.asarray(cluster)
            if cluster.ndim == 0:
                template = self.get_template(cluster)
            else:
                return np.array([self.get_channel(template=t) for t in self.get_template(cluster)])

        template = int(template)
        if template not in self.__tc_cache:
            self.__tc_cache[template] = _get_channel(self, template)

        return self.__tc_cache[template]


def _get_channel(self: KilosortResult, template: int) -> int:
    """
    Get most strong channel index for a template.

    The result may be different when doing the un-whitening or not.

    ## Reference

    * phylib.io.model.TemplateModel._find_best_channels() : they did the un-whitening
    * methods in ecephys_spike_sorting : they use template directly.

    :param template: template index
    :return: channel index
    """
    if template < 0:
        return -1

    data = np.load(self.file.template_file)[template]

    try:
        w = np.load(self.file.whitening_invmat_file)
        data = np.dot(data, w)
    except FileNotFoundError:
        pass

    data = data.astype(np.float32)

    # get max amplitude
    amp = np.max(data, axis=0) - np.min(data, axis=0)

    # get most significant channel
    channel = np.argmax(amp)
    # max_amp = amp[channel]

    return int(channel)
