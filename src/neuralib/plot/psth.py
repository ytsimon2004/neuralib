import numpy as np
import scipy
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

__all__ = [
    'peri_onset_1d',
    'plot_peri_onset_1d'
]


def plot_peri_onset_1d(event_time: np.ndarray,
                       act_time: np.ndarray,
                       act: np.ndarray, *,
                       pre: float = 3,
                       post: float = 5,
                       plot_all: bool = False,
                       bins: int = 100,
                       with_fill_between: bool = True,
                       ax: Axes | None = None,
                       **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Plot peri-event 1D activity

    :param event_time: peri-event time array. `Array[float, P]`
    :param act_time: activity time array. `Array[float, T]`
    :param act: activity array. `Array[float, T]`
    :param pre: peri-event time before
    :param post: peri-event time after
    :param plot_all: plot velocity every laps
    :param bins:  number of bins per trial in a given time bins (peri-left + peri-right)
    :param with_fill_between: fill_between for the sem
    :param ax: ``Axes``
    :param kwargs: additional arguments to ``ax.plot()``
    :return: x (`Array[float, B]`) and average velocity (`Array[float, B]`)
    """
    x = np.linspace(-pre, post, bins)
    v = peri_onset_1d(event_time, act_time, act, bins, pre, post)
    v_mean = np.nanmean(v, axis=0)

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(x, v_mean, **kwargs)
    ax.axvline(0, color="r", linestyle="--", zorder=1)

    if plot_all:
        for vlap in v:
            ax.plot(x, vlap, color='grey', alpha=0.1)

    if with_fill_between:
        v_sem = scipy.stats.sem(v)
        ax.fill_between(x, v_mean + v_sem, v_mean - v_sem, alpha=0.3)

    return x, v_mean


def peri_onset_1d(event_time: np.ndarray,
                  act_time: np.ndarray,
                  act: np.ndarray,
                  bins: int = 100,
                  pre: float = 3,
                  post: float = 5) -> np.ndarray:
    """
    Get Peri-event 1D activity

    :param event_time: peri-event time array. `Array[float, P]`
    :param act_time: activity time array. `Array[float, T]`
    :param act: activity array. `Array[float, T]`
    :param bins: number of bins per trial in a given time bins (peri-left + peri-right)
    :param pre: peri-event time before
    :param post: peri-event time after
    :return: peri-reward activity. `Array[float, [P, B]]`. B = number of bins per trial in a given time bins (peri-left + peri-right)

    """
    ret = np.zeros((len(event_time), bins))
    for i, time in enumerate(event_time):
        left = time - pre
        right = time + post
        time_mask = np.logical_and(left < act_time, act_time < right)
        t = act_time[time_mask]
        v = act[time_mask]

        hist, edg = np.histogram(t, bins, range=(left, right), weights=v)
        occ = np.histogram(t, edg)[0]
        hist /= occ
        hist[np.isnan(hist)] = 0
        ret[i] = hist

    return ret
