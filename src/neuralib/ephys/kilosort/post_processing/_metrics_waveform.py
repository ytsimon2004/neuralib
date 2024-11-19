"""
reference::

    ecephys_spike_sorting.modules.mean_waveforms.waveform_metrics
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from neuralib.ephys.kilosort.result import KilosortResult
from neuralib.util.verbose import fprint
from .waveforms import WaveformResult, get_waveforms

__all__ = [
    'WaveformSpaceMetrics',
    'waveform_amp',
    'waveform_snr',
    'waveform_duration',
    'waveform_half_width',
    'waveform_pt_ratio',
    'waveform_repolarizarion_slope',
    'waveform_recovery_slope',
    'waveform_space_metrics',
]


class WaveformSpaceMetrics(NamedTuple):
    amplitude: float  # uV
    spread: float  # um
    velocity_above: float  # s / m
    velocity_below: float  # s / m

    @classmethod
    def empty(cls) -> WaveformSpaceMetrics:
        return WaveformSpaceMetrics(np.nan, np.nan, np.nan, np.nan)


def waveform_amp(wm: WaveformResult) -> float:
    """Calculate amp of spike waveforms.

    :param wm:
    :return: peak to peak, uV
    """
    assert wm.n_channels == 1, 'should only contain one channel'
    w = wm.waveform[:, 0, :]  # Array[uV:float, N, S]
    w = np.nanmean(w, axis=0)  # Array[uV:float, S]
    return np.max(w) - np.min(w)  # uV


def waveform_snr(wm: WaveformResult) -> float:
    """Calculate SNR of spike waveforms.

    Converted from Matlab by Xiaoxuan Jia

    reference:

    * (Nordhausen et al., 1996; Suner et al., 2005)
    * spikeinterface/qualitymetrics/misc_metrics.py#compute_snrs

    :param wm:
    :return: signal-to-noise ratio for unit (scalar)
    """
    assert wm.n_channels == 1, 'should only contain one channel'
    w = wm.waveform[:, 0, :]  # Array[uV:float, N, S]
    w_bar = np.nanmean(w, axis=0)  # Array[uV:float, S]
    a = np.max(w_bar) - np.min(w_bar)  # uV
    e = w - np.tile(w_bar, (w.shape[0], 1))  # type: np.ndarray # Array[uV:float, N, S]
    return a / (2 * np.nanstd(e.ravel()))


def waveform_duration(wm: WaveformResult) -> float:
    """Duration (in milliseconds) between peak and trough

    :param wm:
    :return: waveform duration in milliseconds
    """
    assert wm.n_channels == 1, 'should only contain one channel'
    waveform = np.mean(wm.waveform[:, 0, :], axis=0)
    t = np.linspace(0, wm.duration * 1000, len(waveform))
    trough_i = np.argmin(waveform)
    peak_i = np.argmax(waveform)

    # to avoid detecting peak before trough
    if waveform[peak_i] > np.abs(waveform[trough_i]):
        i = peak_i + np.argmin(waveform[peak_i:])
        duration = t[i] - t[peak_i]
    else:
        i = trough_i + np.argmax(waveform[trough_i:])
        duration = t[i] - t[trough_i]

    return duration


def waveform_mean_up(wm: WaveformResult, updasample: float = 200 / 82) -> np.ndarray:
    """

    :param wm:
    :param updasample: factor where S' = S * updasample
    :return: Array[uV:float, S'] for major channel
    """
    assert wm.n_channels == 1, 'should only contain one channel'

    from scipy.signal import resample
    new_sample_count = int(wm.n_sample * updasample)
    return resample(np.mean(wm.waveform[:, 0, :], axis=0), new_sample_count)


def waveform_half_width(wm: WaveformResult) -> float:
    """Spike width at half max amplitude

    :param wm:
    :return: waveform halfwidth in milliseconds
    """
    waveform = waveform_mean_up(wm)
    t = np.linspace(0, wm.duration * 1000, len(waveform))
    trough_i = np.argmin(waveform)
    peak_i = np.argmax(waveform)

    try:
        if waveform[peak_i] > np.abs(waveform[trough_i]):
            threshold = waveform[peak_i] * 0.5
            t1 = np.min(np.where(waveform[:peak_i] > threshold)[0])
            t2 = np.min(np.where(waveform[peak_i:] < threshold)[0]) + peak_i
        else:
            threshold = waveform[trough_i] * 0.5
            t1 = np.min(np.where(waveform[:trough_i] < threshold)[0])
            t2 = np.min(np.where(waveform[trough_i:] > threshold)[0]) + trough_i

        return t[t2] - t[t1]
    except ValueError:
        return np.nan


def waveform_pt_ratio(wm: WaveformResult) -> float:
    """Peak-to-trough ratio of 1D waveform

    :param wm:  waveform peak-to-trough ratio
    :return:  waveform peak-to-trough ratio
    """
    assert wm.n_channels == 1, 'should only contain one channel'
    waveform = np.mean(wm.waveform[:, 0, :], axis=0)
    trough_i = np.argmin(waveform)
    peak_i = np.argmax(waveform)
    return np.abs(waveform[peak_i] / waveform[trough_i])


def waveform_repolarizarion_slope(wm: WaveformResult, window=20) -> float:
    """Spike repolarization slope (after maximum deflection point)

    :param wm:
    :param window:
    :return: slope of return to baseline (V / s)
    """
    from scipy.stats import linregress
    waveform = waveform_mean_up(wm)
    xp = np.argmax(np.abs(waveform))
    # invert if we're using the peak
    w = -waveform * (np.sign(waveform[xp]))
    t = np.linspace(0, wm.duration * 1000, len(waveform))
    return linregress(t[xp:xp + window], w[xp:xp + window]).slope


def waveform_recovery_slope(wm: WaveformResult, window=20) -> float:
    """Spike recovery slope (after maximum deflection point)

    :param wm:
    :param window:
    :return: slope of return to baseline (V / s)
    """
    from scipy.stats import linregress
    waveform = waveform_mean_up(wm)
    xp = np.argmax(np.abs(waveform))
    # invert if we're using the peak
    w = -waveform * (np.sign(waveform[xp]))
    pi = np.argmax(waveform[xp:]) + xp
    t = np.linspace(0, wm.duration * 1000, len(waveform))
    return linregress(t[pi:pi + window], w[pi:pi + window]).slope


def waveform_space_metrics(ks_data: KilosortResult, wm: WaveformResult, spread_threshold=0.12) -> WaveformSpaceMetrics:
    """

    :param ks_data:
    :param wm:
    :param spread_threshold: Threshold for computing channel spread of 2D waveform
    :return:
    """
    if wm.n_cluster != 1:
        raise RuntimeError('should only contain one cluster')

    if wm.n_channels < 3:
        raise RuntimeError('neighbor channels too few')

    from scipy.stats import linregress
    waveform = np.mean(wm.waveform, axis=0)  # shape (C, S)
    t = wm.t
    trough_i = np.argmin(waveform, axis=1)
    trough_a = waveform[np.arange(len(trough_i)), trough_i]
    peak_i = np.argmax(waveform, axis=1)
    peak_a = waveform[np.arange(len(trough_i)), peak_i]
    duration = np.abs(t[peak_i] - t[trough_i])
    amplitude = peak_a - trough_a  # all channel amplitude
    channel_x = np.argmax(amplitude)  # peak channel
    channel_a = amplitude[channel_x]  # peak channel amplitude
    act_th = amplitude > channel_a * spread_threshold  # activate channels mask
    pcp = ks_data.channel_pos[ks_data.as_channel_index(wm.channel_list[channel_x])]  # position for peak channel
    acp = ks_data.channel_pos[ks_data.as_channel_index(wm.channel_list[act_th])]  # position for activate channels
    acd = np.sqrt(np.sum((acp - pcp) ** 2, axis=1))  # distance for activate channels
    spd = np.max(acd)  # spread
    chy = ks_data.channel_pos[ks_data.as_channel_index(wm.channel_list[act_th]), 1]
    upc = chy > pcp[1]  # above the peak channel
    dwc = chy < pcp[1]  # below the peak channel

    trough_t = (t[trough_i] - t[trough_i[channel_x]])[act_th]

    if np.count_nonzero(upc) == 0:
        upv = np.nan
    else:
        upp = chy[upc]
        i = np.argsort(upp)
        try:
            upv = linregress(upp[i], trough_t[upc][i]).slope
        except ValueError as e:
            fprint(f'upv: cluster {wm.spike_cluster}, {repr(e)}', vtype='warning')
            upv = np.nan

    if np.count_nonzero(dwc) == 0:
        dwv = np.nan
    else:
        dwp = chy[dwc]
        i = np.argsort(dwp)
        try:
            dwv = linregress(dwp[i], trough_t[dwc][i]).slope
        except ValueError as e:
            fprint(f'dwv: cluster {wm.spike_cluster}, {repr(e)}', vtype='warning')
            dwv = np.nan

    return WaveformSpaceMetrics(channel_a, spd, upv, dwv)


def is_outlier(a: np.ndarray, threshold: float) -> np.ndarray:
    """
    Returns a boolean array with True if points are outliers and False
    otherwise.

    reference::

        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor.


    :param a: An num-observations by num-dimensions array of observations
    :param threshold: The modified z-score to use as a threshold. Observations with
        a modified z-score (based on the median absolute deviation) greater
        than this value will be classified as outliers.
    :return: A num-observations-length boolean array.
    """
    if a.ndim == 1:
        a = a[:, None]

    median = np.median(a, axis=0)
    distance = np.sqrt(np.sum((a - median) ** 2, axis=1))
    median_dis = np.median(distance)
    z_score = 0.6745 * distance / median_dis
    return z_score > threshold


def residual_variance(ks_data: KilosortResult,
                      cluster: int | WaveformResult) -> np.ndarray:
    """
    TODO add reference

    :param ks_data:
    :param cluster:
    :return:
    """
    waveform = None
    if isinstance(cluster, WaveformResult):
        waveform = cluster
        cluster = waveform.spike_cluster
        if cluster is None:
            raise ValueError("waveform does not have major cluster")
        if not isinstance(cluster, int):
            raise ValueError("waveform does not from single cluster")

    cluster = int(cluster)
    channel_index = ks_data.get_channel(cluster=cluster)
    if channel_index is None:
        raise RuntimeError(f'cannot determine the major channel for cluster : {cluster}')
    else:
        channel_index = int(channel_index)
        channel = int(ks_data.as_channel_list(channel_index))

    template = ks_data.template_data[cluster, :, channel_index]  # (sample)
    n_sample_t = len(template)

    if waveform is None:
        duration_ms = 1000 * n_sample_t / ks_data.sample_rate
        waveform = get_waveforms(ks_data, cluster, channel, duration=duration_ms)
    else:
        waveform = waveform.with_channel(channel)

    waveform_data = waveform.waveform[:, 0, :]  # (N, sample)

    n_sample_w = waveform.n_sample
    if n_sample_t != n_sample_w:
        if n_sample_t < n_sample_w:
            m = n_sample_w // 2
            d = n_sample_t // 2
            s = slice(m - d, m + d)
            waveform_data = waveform_data[:, s]
        else:
            m = n_sample_t // 2
            d = n_sample_w // 2
            s = slice(m - d, m + d)
            template = template[s]

    return np.var(waveform_data - template, axis=1)
