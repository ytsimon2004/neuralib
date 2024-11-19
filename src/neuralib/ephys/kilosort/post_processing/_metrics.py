"""
reference::

    ecephys_spike_sorting.modules.quality_metrics.metrics
"""
from typing import NamedTuple

import numpy as np

__all__ = [
    'isi_violation',
    'presence_ratio',
    'firing_rate',
    'amplitude_cutoff',

    'IsiContaminationResult',
    'isi_contamination',
    'auto_correlogram',

    'CcgContaminationResult',
    'ccg_contamination',
]


def isi_violation(spike_train: np.ndarray,
                  duration: float,
                  isi_threshold: float = 1.5,
                  min_isi: float = 0.0) -> tuple[float, int]:
    """Calculate ISI violations for a spike train.

    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    modified by Dan Denman from cortex-lab/sortingQuality GitHub by Nick Steinmetz

    reference:

    * ecephys_spike_sorting/modules/quality_metrics/metrics.py#isi_violations
    * spikeinterface/qualitymetrics/misc_metrics.py#compute_isi_violations

    :param spike_train: spike time in seconds.
    :param duration: data total duration in seconds
    :param isi_threshold: Maximum time (in msec) for ISI violation
    :param min_isi: Minimum time (in msec) for ISI violation
    :return: (rate, count)
    """
    spike_time = spike_train
    dup_spikes = np.nonzero(np.diff(spike_time) <= min_isi)[0]
    spike_time = np.delete(spike_time, dup_spikes + 1)
    isi = np.diff(spike_time)

    n_spike = len(spike_time)
    n_violations = int(np.count_nonzero(isi < isi_threshold / 1000))
    violation_time = 2 * n_spike * (isi_threshold - min_isi) / 1000

    if n_spike == 0:
        return np.nan, 0
    else:
        total_rate = n_spike / duration
        violation_rate = n_violations / violation_time
        fp_ratio = violation_rate / total_rate  # false positive rate
        return fp_ratio, n_violations


def presence_ratio(spike_train: np.ndarray,
                   duration: float,
                   bins=101) -> float:
    """Calculate fraction of time the unit is present within an epoch.

    reference:

    * ecephys_spike_sorting/modules/quality_metrics/metrics.py#presence_ratio
    * spikeinterface/qualitymetrics/misc_metrics.py#compute_presence_ratio

    :param spike_train: spike time in seconds.
    :param duration: data total duration in seconds
    :param bins:
    :return:
    """
    h, _ = np.histogram(spike_train, np.linspace(0, duration, bins))
    return np.count_nonzero(h > 0) / (bins - 1)


def firing_rate(spike_train: np.ndarray,
                duration: float) -> float:
    """Calculate firing rate for a spike train.

    If no temporal bounds are specified, the first and last spike time are used.

    reference:

    * ecephys_spike_sorting/modules/quality_metrics/metrics.py#firing_rate

    :param spike_train: spike time in seconds.
    :param duration: data total duration in seconds
    :return: Firing rate in Hz
    """
    return len(spike_train) / duration


# TODO the result is large different from spikeinterface
def amplitude_cutoff(spike_amp: np.ndarray, bins=500, smooth=11) -> float:
    """Calculate approximate fraction of spikes missing from a distribution of amplitudes

    Assumes the amplitude histogram is symmetric (not valid in the presence of drift)

    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    reference:

    * ecephys_spike_sorting/modules/quality_metrics/metrics.py#amplitude_cutoff
    * spikeinterface/qualitymetrics/misc_metrics.py#compute_amplitudes_cutoff

    :param spike_amp: spike amplitudes
    :param bins:
    :param smooth:
    :return: fraction of missing spikes (0-0.5)
        If more than 50% of spikes are missing, an accurate estimate isn't possible
    """
    from scipy.ndimage.filters import gaussian_filter1d
    # import matplotlib.pyplot as plt

    h, b = np.histogram(spike_amp, bins, density=True)
    bin_size = np.mean(np.diff(b))
    # plt.hist(spike_amp, bins, density=True)

    pdf = gaussian_filter1d(h, smooth)
    # plt.plot(b[:-1], pdf, lw=2)

    peak_i = np.argmax(pdf)
    # plt.axvline(b[peak_i], color='r', lw=2)

    above = np.abs(pdf[peak_i:] - pdf[0])
    g = np.argmin(above) + peak_i
    # plt.axvline(b[g], color='g', lw=2)
    # plt.show()

    fraction_missing = min(0.5, np.sum(pdf[g:]) * bin_size)
    return fraction_missing


class IsiContaminationResult(NamedTuple):
    contamination: float
    """% of spikes following within the refractory period of another"""

    isi: np.ndarray
    """inter spike intervals, `Array[float, N]`"""

    bins: np.ndarray
    """time bins (ms) `Array[float, N+1]`"""

    @property
    def t(self) -> np.ndarray:
        """time (ms), shape `Array[float, N]`"""
        return (self.bins[1:] + self.bins[:-1]) / 2


def isi_contamination(spike_train: np.ndarray,
                      refractory_period: int = 2,
                      max_time: int = 50,
                      bin_size: int = 1) -> IsiContaminationResult:
    """Give the cluster contamination.

    :param spike_train: spike times array, in second
    :param refractory_period: refractory period, msec
    :param max_time: msec
    :param bin_size: msec
    :return ContaminationResult
    """
    from spikeinterface.postprocessing import compute_isi_histograms_from_spiketrain
    fs = 30000
    _max_time = int(max_time * fs / 1000)
    _bin_size = int(bin_size * fs / 1000)
    isi, bins = compute_isi_histograms_from_spiketrain(
        spike_train, max_time=_max_time, bin_size=_bin_size, sampling_f=fs
    )

    rp = int(refractory_period / bin_size)
    contamination = 100 * np.sum(isi[:rp]) / np.sum(isi)

    return IsiContaminationResult(contamination, isi, bins)


class CcgContaminationResult(NamedTuple):
    contamination: float | np.ndarray
    """% of spikes following within the refractory period of another. float or `Array[float, [S, S]]`"""

    auto: np.ndarray  # Array[int, [S, S], N]
    """auto-correlation. `Array[int, N]` or `Array[int, [S, S, N]]`"""

    bins: np.ndarray
    """time bins (ms) `Array[float, N+1]`"""

    @property
    def t(self) -> np.ndarray:
        return (self.bins[1:] + self.bins[:-1]) / 2


def auto_correlogram(spike_time: np.ndarray,
                     max_time: int = 100,
                     bin_size: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """

    :param spike_time:
    :param max_time:
    :param bin_size:
    :return: (auto, bins), where bins in ms
    """
    fs = 30000
    max_time = int(max_time * fs / 1000)
    bin_size = int(bin_size * fs / 1000)
    # spikeinterface compute_autocorrelogram_* newer version remove parameter fs,
    # and we need to make bins by ourselves because they removed it
    bins = np.arange(-max_time, max_time + bin_size, bin_size) * 1000 / fs  # ms

    from spikeinterface.postprocessing import compute_autocorrelogram_from_spiketrain as cafs

    spike_time = spike_time * fs
    try:
        auto = cafs(spike_time, max_time, bin_size)
    except TypeError as e:
        raise RuntimeError('numba is not installed') from e

    return auto, bins


def ccg_contamination(*spike_time: np.ndarray,
                      max_time: int = 100,
                      bin_size: int = 1) -> CcgContaminationResult:
    """Give the cluster contamination.

    reference: Kilosort2/postProcess/set_cutoff.m

    :param spike_time: spike times array, in second
    :param max_time: msec
    :param bin_size: msec
    :return ContaminationResult
    """
    s = len(spike_time)
    if s == 0:
        raise RuntimeError('empty list')
    elif s == 1 and len(spike_time[0]) <= 10:
        return CcgContaminationResult(np.nan, np.empty((0,)), np.empty((0, 2)))

    from ._metrics_ccg import ccg

    fs = 30000
    max_time = int(max_time * fs / 1000)
    bin_size = int(bin_size * fs / 1000)
    # spikeinterface compute_autocorrelogram_* newer version remove parameter fs,
    # and we need to make bins by ourselves because they removed it
    bins = np.arange(-max_time, max_time + bin_size, bin_size) * 1000 / fs  # ms

    if s == 1:
        from spikeinterface.postprocessing import compute_autocorrelogram_from_spiketrain as cafs
        spike_time = spike_time[0]
        t = spike_time[-1] - spike_time[0]
        a = len(spike_time) ** 2

        try:
            auto = cafs(spike_time * fs, max_time, bin_size)
        except TypeError as e:
            raise RuntimeError('numba is not installed') from e

        result = ccg(t, a, auto, bins)
        return CcgContaminationResult(100 * result.q, result.auto, bins)
    else:
        from spikeinterface.postprocessing import compute_crosscorrelogram_from_spiketrain as ccfs

        nbins = int(2 * max_time / bin_size)
        cont = np.zeros((s, s), dtype=float)
        auto = np.zeros((s, s, nbins), dtype=float)

        for a in range(s):
            for b in range(s):
                t = max(spike_time[a][-1], spike_time[b][-1]) - min(spike_time[a][0], spike_time[b][0])
                ab = len(spike_time[a]) * len(spike_time[b])

                try:
                    _auto = ccfs(spike_time[a] * fs, spike_time[b] * fs, max_time, bin_size)
                except TypeError as e:
                    raise RuntimeError('numba is not installed') from e

                result = ccg(t, ab, _auto, bins)
                cont[a, b] = 100 * result.q
                auto[a, b] = result.auto

        return CcgContaminationResult(cont, auto, bins)
