from typing import NamedTuple

import numpy as np

from neuralib.ephys.kilosort.result import KilosortResult


class PCFeatureMetrics(NamedTuple):
    pass


def pc_feature(ks_data: KilosortResult,
               nch_compare: int = 13,
               max_spikes_unit: int = 500,
               max_spikes_nn: int = 10000,
               n_neighbors: int = 4):
    if nch_compare % 2 != 1:
        raise ValueError(f'nch_compare should be odd : {nch_compare}')

    half_spread = int((nch_compare - 1) / 2)

    clusters = np.unique(ks_data.spike_cluster)
    templates = np.unique(ks_data.spike_template)

    template_peak = np.zeros((len(templates),), dtype='uint16')  # template peak channel
    cluster_peak = np.zeros((len(clusters),), dtype='uint16')  # cluster peak channel

    for i, t in enumerate(templates):
        pc_max = np.argmax(np.mean(ks_data.pc_features[ks_data.spike_template == t, 0, :], axis=0))
        template_peak[i] = ks_data.pc_feature_index[t, pc_max]

    for i, c in enumerate(clusters):
        t = np.unique(ks_data.spike_template[ks_data.spike_cluster == c])
        p = np.nonzero(np.isin(templates, t))[0]
        cluster_peak[i] = np.median(template_peak[p])

    for i, c in enumerate(clusters):
        _cluster_metrics(ks_data, i, c, cluster_peak, nch_compare, max_spikes_unit, max_spikes_nn, n_neighbors)


def _cluster_metrics(ks_data: KilosortResult,
                     i: int,
                     cluster: int,
                     cluster_peak: np.ndarray,
                     nch_compare: int = 13,
                     max_spikes_unit: int = 500,
                     max_spikes_nn: int = 10000,
                     n_neighbors: int = 4):
    if nch_compare % 2 != 1:
        raise ValueError(f'nch_compare should be odd : {nch_compare}')

    half_spread = int((nch_compare - 1) / 2)

    peak_channel = cluster_peak[i]
    n_spikes = np.count_nonzero(ks_data.spike_cluster == cluster)

    half_spread_down = peak_channel if peak_channel < half_spread else half_spread

    x_feature_i = np.max(ks_data.pc_feature_index)
    half_spread_up = x_feature_i - peak_channel if peak_channel + half_spread_down > x_feature_i else half_spread
