import numpy as np
from typing_extensions import overload

from neuralib.ephys.kilosort.result import KilosortResult
from . import _metrics
from .utils import iter_spike_time

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


# TODO check all joblib call

@overload
def isi_violation(ks_data: KilosortResult,
                  cluster_list: np.ndarray, *,
                  isi_threshold: float = ...,
                  min_isi: float = ...) -> np.ndarray:
    pass


def isi_violation(ks_data: KilosortResult,
                  cluster_list: np.ndarray,
                  **kwargs) -> np.ndarray:
    from joblib import Parallel, delayed
    duration = ks_data.time_duration
    return Parallel()(
        delayed(_metrics.isi_violation)(spike_trains, duration, **kwargs)
        for spike_trains in iter_spike_time(ks_data, cluster_list)
    )


@overload
def presence_ratio(ks_data: KilosortResult,
                   cluster_list: np.ndarray, *,
                   bins: int = ...) -> np.ndarray:
    pass


def presence_ratio(ks_data: KilosortResult,
                   cluster_list: np.ndarray,
                   **kwargs) -> np.ndarray:
    from joblib import Parallel, delayed
    duration = ks_data.time_duration
    return Parallel()(
        delayed(_metrics.presence_ratio)(spike_trains, duration, **kwargs)
        for spike_trains in iter_spike_time(ks_data, cluster_list)
    )


def firing_rate(ks_data: KilosortResult,
                cluster_list: np.ndarray) -> np.ndarray:
    from joblib import Parallel, delayed
    duration = ks_data.time_duration
    return Parallel()(
        delayed(_metrics.firing_rate)(spike_trains, duration)
        for spike_trains in iter_spike_time(ks_data, cluster_list)
    )


@overload
def amplitude_cutoff(ks_data: KilosortResult,
                     cluster_list: np.ndarray, *,
                     bins: int = ...,
                     smooth: int = ...) -> np.ndarray:
    pass


def amplitude_cutoff(ks_data: KilosortResult,
                     cluster_list: np.ndarray,
                     **kwargs) -> np.ndarray:
    from joblib import Parallel, delayed
    duration = ks_data.time_duration
    return Parallel()(
        delayed(_metrics.amplitude_cutoff)(spike_trains, duration, **kwargs)
        for spike_trains in iter_spike_time(ks_data, cluster_list)
    )


IsiContaminationResult = _metrics.IsiContaminationResult


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
    from joblib import Parallel, delayed
    return Parallel()(
        delayed(_metrics.isi_contamination)(spike_trains, **kwargs)
        for spike_trains in iter_spike_time(ks_data, cluster_list)
    )


CcgContaminationResult = _metrics.CcgContaminationResult


@overload
def ccg_contamination(ks_data: KilosortResult,
                      cluster_list: np.ndarray = None, *,
                      max_time: int = ...,
                      bin_size: int = ...) -> list[CcgContaminationResult]:
    pass


def ccg_contamination(ks_data: KilosortResult,
                      cluster_list: np.ndarray = None,
                      **kwargs) -> list[CcgContaminationResult]:
    from joblib import Parallel, delayed
    duration = ks_data.time_duration
    return Parallel()(
        delayed(_metrics.ccg_contamination)(spike_trains, duration, **kwargs)
        for spike_trains in iter_spike_time(ks_data, cluster_list)
    )
