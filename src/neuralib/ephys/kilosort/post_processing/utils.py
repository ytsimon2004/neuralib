import numpy as np

from neuralib.ephys.kilosort.result import KilosortResult

__all__ = [
    'next_cluster_id',
    'template_peak_channel'
]


def next_cluster_id(ks_data: KilosortResult) -> int:
    return np.max(ks_data.spike_cluster) + 1


def template_peak_channel(template: np.ndarray) -> int | np.ndarray:
    """

    :param template: Array[float, [S, C]] or Array[float, [U, S, C]]
    :return: channel index (array if template.ndim==3)
    """
    if template.ndim == 2:  # (S, C)
        return int(np.argmax(np.max(template, axis=0) - np.min(template, axis=0)))
    elif template.ndim == 3:  # (U, S, C)
        return np.argmax(np.max(template, axis=1) - np.min(template, axis=1), axis=1)
    else:
        raise ValueError('wrong dimension')
