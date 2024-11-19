"""
reference::

    ecephys_spike_sorting.modules.quality_metrics.metrics
"""

import numpy as np

__all__ = [
    'isi_violation',
    'presence_ratio',
    'firing_rate',
    'amplitude_cutoff',
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
