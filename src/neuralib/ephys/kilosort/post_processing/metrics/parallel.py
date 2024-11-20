import numpy as np
from typing_extensions import overload

from neuralib.ephys.kilosort.post_processing.utils import iter_spike_time
from neuralib.ephys.kilosort.result import KilosortResult
from . import metrics

__all__ = [
    'isi_violation',
    'presence_ratio',
    'firing_rate',
    'amplitude_cutoff',

    'IsiContaminationResult',
    'isi_contamination',

    'CcgContaminationResult',
    'ccg_contamination',
]



@overload
def isi_violation(ks_data: KilosortResult,
                  cluster_list: np.ndarray, *,
                  isi_threshold: float = ...,
                  min_isi: float = ...) -> np.ndarray:
    pass


def isi_violation(ks_data: KilosortResult,
                  cluster_list: np.ndarray,
                  **kwargs) -> np.ndarray:
    """
    Calculate ISI violations for clusters.

    :param ks_data:
    :param cluster_list: cluster `Array[int, C]`
    :param kwargs: pass to metrics.isi_violation
    :return: `Array[float, C]`
    :seealso: metrics.isi_violation
    """
    from joblib import Parallel, delayed
    duration = ks_data.time_duration
    return np.array(Parallel()(
        delayed(metrics.isi_violation)(spike_trains, duration, **kwargs)
        for spike_trains in iter_spike_time(ks_data, cluster_list)
    ))


@overload
def presence_ratio(ks_data: KilosortResult,
                   cluster_list: np.ndarray, *,
                   bins: int = ...) -> np.ndarray:
    pass


def presence_ratio(ks_data: KilosortResult,
                   cluster_list: np.ndarray,
                   **kwargs) -> np.ndarray:
    """
    Calculate fraction of time the unit is present within an epoch.

    :param ks_data:
    :param cluster_list: cluster `Array[int, C]`
    :param kwargs: pass to metrics.presence_ratio
    :return: `Array[float, C]`
    :seealso: metrics.presence_ratio
    """
    from joblib import Parallel, delayed
    duration = ks_data.time_duration
    return np.array(Parallel()(
        delayed(metrics.presence_ratio)(spike_trains, duration, **kwargs)
        for spike_trains in iter_spike_time(ks_data, cluster_list)
    ))


def firing_rate(ks_data: KilosortResult,
                cluster_list: np.ndarray) -> np.ndarray:
    """
    Calculate firing rate for clusters

    :param ks_data:
    :param cluster_list: cluster `Array[int, C]`
    :return: `Array[float, C]`
    :seealso: metrics.firing_rate
    """
    from joblib import Parallel, delayed
    duration = ks_data.time_duration
    return np.array(Parallel()(
        delayed(metrics.firing_rate)(spike_trains, duration)
        for spike_trains in iter_spike_time(ks_data, cluster_list)
    ))


@overload
def amplitude_cutoff(ks_data: KilosortResult,
                     cluster_list: np.ndarray, *,
                     bins: int = ...,
                     smooth: int = ...) -> np.ndarray:
    pass


def amplitude_cutoff(ks_data: KilosortResult,
                     cluster_list: np.ndarray,
                     **kwargs) -> np.ndarray:
    """
    Calculate approximate fraction of spikes missing from a distribution of amplitudes

    :param ks_data:
    :param cluster_list: cluster `Array[int, C]`
    :param kwargs: pass to metrics.amplitude_cutoff
    :return: `Array[float, C]`
    :seealso: metrics.amplitude_cutoff
    """
    from joblib import Parallel, delayed
    duration = ks_data.time_duration
    return np.array(Parallel()(
        delayed(metrics.amplitude_cutoff)(spike_trains, duration, **kwargs)
        for spike_trains in iter_spike_time(ks_data, cluster_list)
    ))


IsiContaminationResult = metrics.IsiContaminationResult


@overload
def isi_contamination(ks_data: KilosortResult,
                      cluster_list: np.ndarray, *,
                      refractory_period: int = ...,
                      max_time: int = ...,
                      bin_size: int = ...) -> list[IsiContaminationResult]:
    pass


def isi_contamination(ks_data: KilosortResult,
                      cluster_list: np.ndarray,
                      **kwargs) -> list[IsiContaminationResult]:
    """
    Calculate inter-spike interval contamination for clusters.

    :param ks_data:
    :param cluster_list:
    :param kwargs: pass to metrics.isi_contamination
    :return:
    :seealso: metrics.isi_contamination
    """
    from joblib import Parallel, delayed
    return Parallel()(
        delayed(metrics.isi_contamination)(spike_trains, **kwargs)
        for spike_trains in iter_spike_time(ks_data, cluster_list)
    )


CcgContaminationResult = metrics.CcgContaminationResult


@overload
def ccg_contamination(ks_data: KilosortResult,
                      cluster_list: np.ndarray = None, *,
                      max_time: int = ...,
                      bin_size: int = ...) -> list[CcgContaminationResult]:
    pass


def ccg_contamination(ks_data: KilosortResult,
                      cluster_list: np.ndarray = None,
                      **kwargs) -> list[CcgContaminationResult]:
    """
    Calculate the auto-cross-correlation contamination for clusters.

    :param ks_data:
    :param cluster_list:
    :param kwargs: pass to metrics.ccg_contamination
    :return:
    :seealso: metrics.ccg_contamination
    """
    from joblib import Parallel, delayed
    duration = ks_data.time_duration
    return Parallel()(
        delayed(metrics.ccg_contamination)(spike_trains, duration, **kwargs)
        for spike_trains in iter_spike_time(ks_data, cluster_list)
    )
