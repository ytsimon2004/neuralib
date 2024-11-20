from typing import NamedTuple

import numpy as np
from scipy.special import erf

__all__ = ['ccg', 'CCG']


class CCG(NamedTuple):
    auto: np.ndarray
    bins: np.ndarray
    Q00: float  # normalized mean firing rate within (-50] + [50~) ms
    Q01: float  # normalized mean firing rate within [-50, -10) + (10, 50] ms
    Qi: np.ndarray  # Array[float, B=10], normalized mean firing rate within [-B, B] - {0} ms
    Ri: np.ndarray  # Array[float, B], likelihood

    @property
    def t(self) -> np.ndarray:
        return (self.bins[1:] + self.bins[:-1]) / 2

    @property
    def q(self) -> float:
        """ measure of refractoriness

        https://github.com/MouseLand/Kilosort/blob/main/kilosort/CCG.py#L70

        :return: Q12 (kilosort)
        """
        ret = np.min(self.Qi) / max(self.Q00, self.Q01)
        if np.isnan(ret):
            # https://github.com/MouseLand/Kilosort/blob/main/postProcess/set_cutoff.m#L72
            return 1.0
        else:
            return ret

    @property
    def r(self) -> float:
        """the estimated probability that any of the center bins are refractory.

        https://github.com/MouseLand/Kilosort/blob/main/kilosort/CCG.py#L71

        :return: R12 (kilosort)
        """
        return np.min(self.Ri)

    @property
    def rt(self) -> int:
        return int(np.argmin(self.Ri))

    def is_refactory(self, acg_threshold=0.2) -> bool:
        # https://github.com/MouseLand/Kilosort/blob/main/kilosort/CCG.py#L82
        return self.q < acg_threshold and self.r < 0.2

    def cross_refactory(self, ccg_threshold=0.25) -> bool:
        # https://github.com/MouseLand/Kilosort/blob/main/kilosort/CCG.py#L83
        return self.q < ccg_threshold and self.r < 0.05


def ccg(t: float, a: int, auto: np.ndarray, bins: np.ndarray) -> CCG:
    """

    reference:

    * kilosort 3

      https://github.com/MouseLand/Kilosort/blob/kilosort3/postProcess/ccg.m

    * kilosort 4

      https://github.com/MouseLand/Kilosort/blob/main/kilosort/CCG.py#L39

    :param t: duration time in sec
    :param a: square of total spikes
    :param auto: auto-correlogram
    :param bins: time bins in ms
    :return:
    """
    if len(auto) + 1 != len(bins):
        raise ValueError(f'size mismatch auto : {auto.shape} + 1 != bins : {bins.shape}')

    n4 = len(bins) - 1
    n2 = n4 // 2
    n3 = 3 * n4 // 2
    dt = np.mean(np.diff(bins))  # ms
    b10 = int(10 / dt)  # bin number within 10 ms
    b50 = int(50 / dt)  # bin number within 50 ms
    if a == 0:
        nana = np.full((b10,), np.nan)
        return CCG(auto, bins, np.nan, np.nan, nana, nana)

    # this index range corresponds to the CCG shoulders
    i1 = np.concatenate([np.arange(0, n2), np.arange(n3, n4)])
    # these indices are the narrow, immediate shoulders
    i2 = np.arange(n2 - b50, n2 - b10)
    i3 = np.arange(n2 + b10 + 1, n2 + b50 + 1)

    # normalize the shoulders by what's expected from the mean firing rates
    # a non-refractive poisson process should yield 1
    f = dt * a / t / 1000
    q0 = np.sum(auto[i1]) / len(i1) / f
    q1 = max(np.sum(auto[i2]) / len(i2) / f, np.sum(auto[i3]) / len(i3) / f)

    # average rate, take the biggest shoulder
    r0 = max(np.mean(auto[i1]), np.mean(auto[i2]), np.mean(auto[i3]))

    # test the probability that a central area in the auto-correlogram might be refractory
    # test increasingly larger areas of the central CCG
    qi = np.zeros((b10,))
    ri = np.zeros((b10,))
    for i in range(1, b10 + 1):
        #  for this central range of the CCG
        ir = np.arange(n2 - i, n2 + 1 + i)  # [-i, i] ms
        assert len(ir) == 2 * i + 1, f'{len(ir)=} != {2 * i + 1}'

        # compute the same normalized ratio as above. this should be 1 if there is no refractoriness
        si = np.sum(auto[ir]) - auto[n2]
        qi[i - 1] = si / (2 * i * f)  # save the normalized probability

        # this is tricky: we approximate the Poisson likelihood with a gaussian of equal mean and variance
        # that allows us to integrate the probability that we would see <N spikes in the center of the
        # cross-correlogram from a distribution with mean `r0 * i` spikes
        la = r0 * i  # Poisson λ
        if la > 0:
            # https://en.wikipedia.org/wiki/Error_function#Applications
            # Pr[X <= L] = 1/2 + 1/2 erf((L - μ) / (sqrt(2) σ) )
            # Poisson la = μ; la = σ^2
            ri[i - 1] = 0.5 + erf((si / 2 - la) / np.sqrt(2 * la)) / 2
        else:
            ri[i - 1] = 1
    return CCG(auto, bins, q0, q1, qi, ri)
