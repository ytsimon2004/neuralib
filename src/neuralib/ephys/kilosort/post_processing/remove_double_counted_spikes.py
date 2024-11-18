"""
ecephys_spike_sorting/modules/kilosort_postprocessing/postprocessing.py
"""
from typing import NamedTuple

import numpy as np

from neuralib.ephys.kilosort.result import KilosortResult
from .utils import template_peak_channel, next_cluster_id

__all__ = [
    'RemoveDoubleSpikeResult',
    'remove_double_counted_spikes'
]


class RemoveDoubleSpikeResult(NamedTuple):
    noise_cluster: int
    overlap_matrix: np.ndarray  # shape (clusters, clusters).
    """Matrix indicating number of spikes removed for each pair of clusters"""
    cluster_list: np.ndarray
    within_unit_overlap_window: float
    between_unit_overlap_window: float
    between_unit_channel_distance: int | None


def remove_double_counted_spikes(ks_data: KilosortResult,
                                 within_unit_overlap_window: float = 0.166,
                                 between_unit_overlap_window: float = 0.166,
                                 between_unit_channel_distance: int | None = 5 * 15) -> RemoveDoubleSpikeResult:
    """
    Remove putative double-counted spikes from Kilosort outputs.

    The removed spike will be assigned to a new cluster ID.

    :param ks_data:
    :param within_unit_overlap_window: Time window (ms) for removing overlapping spikes for one unit.
    :param between_unit_overlap_window: Time window (ms) for removing overlapping spikes between two units.
    :param between_unit_channel_distance: Number of um (above and below peak channel) to search for
        overlapping spikes
    :return:
    """
    cluster_list = np.unique(ks_data.spike_cluster)
    template_index = ks_data.get_template(cluster_list)
    peak_channels = template_peak_channel(ks_data.template_data)[template_index]
    channel_y = ks_data.channel_pos[peak_channels, 1]
    assert len(peak_channels) == len(channel_y)
    overlap_matrix = np.zeros((len(peak_channels), len(peak_channels)), dtype=int)

    within_unit_overlap_sample = within_unit_overlap_window * ks_data.sample_rate / 1000
    between_unit_overlap_sample = between_unit_overlap_window * ks_data.sample_rate / 1000

    # Removing within-unit overlapping spikes...
    spikes_to_remove = []
    for i, c1 in enumerate(cluster_list):
        s = np.nonzero(ks_data.spike_cluster == c1)[0]
        r = find_within_unit_overlap(ks_data.spike_timestep[s], within_unit_overlap_sample)
        overlap_matrix[i, i] = len(r)
        spikes_to_remove.extend(s[r])
    spikes_to_remove = np.array(spikes_to_remove)
    c = remove_spikes(ks_data, spikes_to_remove)

    bucd = between_unit_channel_distance
    # Removing between-unit overlapping spikes...
    spikes_to_remove = []
    for i, c1 in enumerate(cluster_list):
        s1 = np.nonzero(ks_data.spike_cluster == c1)[0]
        st1 = ks_data.spike_timestep[s1]

        for j, c2 in enumerate(cluster_list):
            if j > i and (bucd is None or np.abs(channel_y[i] - channel_y[j]) <= bucd):
                s2 = np.nonzero(ks_data.spike_cluster == c2)[0]
                st2 = ks_data.spike_timestep[s2]
                r1, r2 = find_between_unit_overlap(st1, st2, between_unit_overlap_sample)
                overlap_matrix[i, j] = len(r1) + len(r2)
                # overlap_matrix[j, i] = overlap_matrix[i, j]
                spikes_to_remove.extend(s1[r1])
                spikes_to_remove.extend(s2[r2])
    spikes_to_remove = np.array(spikes_to_remove)
    remove_spikes(ks_data, spikes_to_remove, c)

    return RemoveDoubleSpikeResult(
        c, overlap_matrix, cluster_list,
        within_unit_overlap_window, between_unit_overlap_window, between_unit_channel_distance
    )


def find_within_unit_overlap(spike_train: np.ndarray, overlap_window: float = 5) -> np.ndarray:
    return np.nonzero(np.diff(spike_train) < overlap_window)[0]


def find_between_unit_overlap(s1: np.ndarray,
                              s2: np.ndarray,
                              overlap_window: float = 5) -> tuple[np.ndarray, np.ndarray]:
    st = np.concatenate((s1, s2))
    si = np.concatenate((np.arange(len(s1)), np.arange(len(s2))))
    sc = np.concatenate((np.zeros_like(s1), np.ones_like(s2)), dtype=int)
    ordering = np.argsort(st)
    st = st[ordering]
    si = si[ordering][1:]
    sc = sc[ordering][1:]
    r = np.diff(st) < overlap_window
    r1 = si[np.logical_and(r, sc == 0)]
    r2 = si[np.logical_and(r, sc == 1)]
    return r1, r2


def remove_spikes(ks_data: KilosortResult, i: np.ndarray, c: int = None) -> int:
    if c is None:
        c = next_cluster_id(ks_data)
    ks_data.spike_cluster[i] = c
    return c
