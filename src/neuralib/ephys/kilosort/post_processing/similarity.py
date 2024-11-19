from collections.abc import Iterator

import numpy as np

from neuralib.ephys.kilosort.files import KilosortFiles
from neuralib.ephys.kilosort.result import KilosortResult

__all__ = [
    'similarity_group',
    'foreach_similar_group',
    'foreach_similar_clusters'
]


def similarity_group(ks_data: KilosortFiles | KilosortResult, similar_threshold: float = 0.8):
    """
    grouping clusters (base on template ID) by their similar index.

    :param ks_data:
    :param similar_threshold:
    :return: similarity_group `Array[group:int, U]`,
        where group is a discrete value, and -1 means orphan waveform.
    """
    if not (0 <= similar_threshold <= 1):
        raise ValueError(f'illegal threshold value : {similar_threshold}')

    if isinstance(ks_data, KilosortResult):
        ks_data = ks_data.file

    m = np.load(ks_data.similar_templates_file)  # Array[float32, [U, U]]
    t = m.shape[0]

    cluster_group = np.full((t,), -1, dtype=int)
    group_counter = 0
    for c in range(t):
        sc = np.nonzero(m[c] >= similar_threshold)[0]
        sc = sc[np.argsort(-m[c, sc])]

        if len(sc) <= 1:
            continue

        group = cluster_group[sc]
        if np.all(group == -1):
            cluster_group[sc] = group_counter
            group_counter += 1
        else:
            group = tuple(np.unique(group))
            if len(group) == 1:  # (G, G) => (G, G)
                pass
            elif group[0] == -1 and len(group) == 2:  # (-1, G) => (G, G)
                assert group[1] >= 0
                cluster_group[sc] = group[1]
            else:  # (G, G', ...) => (G", ...)
                cluster_group[sc] = group_counter
                for g in group:  # union G'
                    if g >= 0:
                        cluster_group[cluster_group == g] = group_counter
                group_counter += 1

    return cluster_group


def foreach_similar_group(group: np.ndarray) -> Iterator[np.ndarray]:
    """

    :param group: similarity_group `Array[group:int, U]`
    :return: iterator of template index array `Array[U, N]` for each group
    """
    for g in range(np.max(group) + 1):
        x = np.nonzero(group == g)[0]
        if len(x) > 0:
            yield x


def foreach_similar_clusters(ks_data: KilosortFiles | KilosortResult, group: np.ndarray) -> Iterator[np.ndarray]:
    """

    :param ks_data:
    :param group: similarity_group `Array[group:int, U]`
    :return: iterator of cluster list `Array[cluster:int, N]` for each group
    """
    if isinstance(ks_data, KilosortFiles):
        ks_data = ks_data.result()

    for g in range(np.max(group) + 1):
        tt = np.nonzero(group == g)[0]
        if len(tt) > 0:
            ss = [np.nonzero(ks_data.spike_template == t)[0] for t in tt]
            cc = [np.unique(ks_data.spike_cluster[s]) for s in ss]
            yield np.unique(np.concatenate(cc))
