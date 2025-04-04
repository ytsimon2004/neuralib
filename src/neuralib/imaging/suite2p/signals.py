from typing import Literal, NamedTuple

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, minimum_filter1d, maximum_filter1d

from .core import Suite2PResult, SIGNAL_TYPE

__all__ = [
    'get_neuron_signal',
    'normalize_signal',
    'normalize_signal_factor',
    #
    'DFFSignal',
    'dff_signal',
    'calc_signal_baseline',
    #
    'sync_s2p_rigevent',
]

BASELINE_METHOD = Literal['maximin', 'constant', 'constant_prctile']


def get_neuron_signal(s2p: Suite2PResult,
                      n: int | np.ndarray | list[int] | None = None,
                      *,
                      signal_type: SIGNAL_TYPE = 'df_f',
                      normalize: bool = True,
                      dff: bool = True,
                      correct_neuropil: bool = True,
                      method: BASELINE_METHOD = 'maximin') -> tuple[np.ndarray, np.ndarray]:
    """
    Select neuronal signals for analysis. For single cell (F,) OR multiple cells (N, F)

    :param s2p: suite 2p result
    :param n: neuron index (`int`) or index arraylike (`Array[int, N]`). If None, then use all neurons
    :param signal_type: signal type. :data:`~neuralib.imaging.suite2p.core.SIGNAL_TYPE` {'df_f', 'spks'}
    :param normalize: 01 normalization for each neuron
    :param dff: normalize to the baseline fluorescence changed (dF/F)
    :param correct_neuropil: do the neuropil correction
    :param method: baseline calculation method {'maximin', 'constant', 'constant_prctile'}
    :return: tuple with (signal, baseline signal). `Array[float, F|[N,F]]`
    """
    if n is None:
        n = list(range(s2p.n_neurons))

    f = s2p.f_raw[n]
    fneu = s2p.f_neu[n]

    if signal_type == 'df_f':

        if dff:
            s1 = dff_signal(f, fneu, s2p, correct_neuropil, method).dff
            s2 = np.full_like(s1, 0, dtype=int)
            if normalize:
                o, f = normalize_signal_factor(s1)
                s1 = (s1 - o) / f

        else:
            fcorr = f - 0.7 * fneu
            sig_baseline = s2p.signal_baseline
            window = int(s2p.window_baseline * s2p.fs)
            s1 = fcorr
            # baseline not a constant value, due to calcium decay or motion drift
            s2 = _maximin_filter(fcorr, sig_baseline, window)
            if normalize:
                o, f = normalize_signal_factor(s1)
                s1 = (s1 - o) / f
                ob, fb = normalize_signal_factor(s2)
                s2 = (s2 - ob) / fb

    elif signal_type == 'spks':
        s1 = s2p.spks[n]
        s2 = np.full_like(s1, 0, dtype=int)

        if normalize:
            s1 = normalize_signal(s1)

    else:
        raise ValueError('specify signal type either in df_f or spks')

    return s1, s2


def normalize_signal(s: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    do 0 1 normalization

    :param s: signal
    :param axis: axis
    :return: normalized signal
    """
    o, f = normalize_signal_factor(s, axis)
    return (s - o) / f


def normalize_signal_factor(s: np.ndarray, axis: int = -1) -> tuple[np.ndarray, np.ndarray]:
    return np.min(s, axis=axis, keepdims=True), \
        np.max(s, axis=axis, keepdims=True) - np.min(s, axis=axis, keepdims=True)


# ============ #
# DF/F Compute #
# ============ #

class DFFSignal(NamedTuple):
    """Container for dF/F signal processing.

    `Dimension parameters`:

        N = number of neurons

        F = number of frame

    For single cell (F,) OR multiple cells (N, F)

    """

    s2p: Suite2PResult
    """Suite2PResult"""
    f: np.ndarray
    """Fluorescence intensity. `Array[float, F | [N,F]]`"""
    fneu: np.ndarray
    """Neuropil intensity. `Array[float, F | [N,F]]`"""
    fcorr: np.ndarray
    """Fluorescence corrected by fneu that used for dff calculation. `Array[float, F | [N,F]]`"""
    f0: np.ndarray
    """Background Fluorescence. `Array[float, F | [N,F]]`"""
    dff: np.ndarray
    """dff after f0 normalization. `Array[float, F | [N,F]]`"""

    @property
    def dff_baseline(self) -> np.ndarray:
        """baseline of dff, supposed to be 0"""
        return np.full_like(self.dff, 0)

    @property
    def baseline_fluctuation(self) -> np.ndarray:
        """get the fluctuation of the fneu signal. `Array[float, F | [N,F]]`

        Perhaps not fully corrected with physiological reason.
        **used to get the baseline std (i.e., trial reliability metric)**
        """
        fneu_corr = 0.3 * self.fneu
        fneu_bas = calc_signal_baseline(fneu_corr, self.s2p, method='maximin')
        return 100 * ((fneu_corr - fneu_bas) / fneu_bas)


def dff_signal(f: np.ndarray,
               fneu: np.ndarray,
               s2p: Suite2PResult,
               correct_neuropil: bool = True,
               method: BASELINE_METHOD = 'maximin') -> DFFSignal:
    """
    df_f signal normalization container

    :param f: Fluorescence intensity. `Array[float, F | [N,F]]`
    :param fneu: Neuropil intensity. `Array[float, F | [N,F]]`
    :param s2p: ``Suite2PResult``
    :param correct_neuropil: whether do the subtraction using neuropil
    :param method: {'maximin', 'constant', 'constant_prctile'}

    :return: DFFSignal
    """

    fcorr = f - 0.7 * fneu if correct_neuropil else f

    f0 = calc_signal_baseline(fcorr, s2p, method)
    dff = 100 * ((fcorr - f0) / f0)
    return DFFSignal(s2p, f, fneu, fcorr, f0, dff)


def calc_signal_baseline(signal: np.ndarray,
                         s2p: Suite2PResult,
                         method: BASELINE_METHOD = 'maximin') -> np.ndarray:
    """
    f0 calculation

    .. seealso:: source code from suite2p: ``suite2p.extraction.dcnv.preprocess``

    :param signal: signal activity. i.e., fcorr
    :param s2p: :class:`~neuralib.imaging.suite2p.core.Suite2PResult`
    :param method: BASELINE_METHOD
    :return:
        f0 baseline for the df/f calculation
    """

    if method == 'maximin':
        sig_baseline = s2p.signal_baseline
        window = int(s2p.window_baseline * s2p.fs)
        f0 = _maximin_filter(signal, sig_baseline, window)

    elif method == 'constant':
        signal = gaussian_filter(signal, [0., s2p.signal_baseline])
        f0 = np.min(signal, axis=-1)

    elif method == 'constant_prctile':
        sig_prc = np.percentile(signal, s2p.prctile_baseline, axis=-1)
        f0 = np.expand_dims(sig_prc, axis=-1)

    else:
        raise TypeError(f'unknown method{method}')

    return f0


def _maximin_filter(signal: np.ndarray, kernel: float, size: int) -> np.ndarray:
    """
    max/min method to calculate the signal baseline.
    **Note that this method is sensitive to the kernel size

    :param signal:
    :param kernel: standard deviation for Gaussian kernel
    :param size: length along which to calculate 1D minimum
    :return:
    """
    signal = gaussian_filter1d(signal, kernel, axis=-1)
    signal = minimum_filter1d(signal, size, axis=-1)
    baseline = maximum_filter1d(signal, size, axis=-1)

    return baseline


# ================= #
# Sync Pulse Number #
# ================= #

def sync_s2p_rigevent(image_time: np.ndarray,
                      s2p: Suite2PResult,
                      plane: int | None = 0) -> np.ndarray:
    r"""

    Do the alignment to make the shape the same in hardware pulses `(P,)` & suite2p output files `(F,)`

    \* ASSUME both recordings are sync

    \* ASSUME sequential scanning pattern. i.e., 0, 1, 2, 3 ETL scanning order

    :param image_time: imaging time hardware pulses. `Array[float, P]`
    :param s2p: :class:`~neuralib.imaging.suite2p.core.Suite2PResult`
    :param plane: number of optical plane
    :return: image_time: with same len as s2p results `Array[float, F]`
    """
    n_plane = s2p.n_plane
    image_fr = s2p.fs

    if plane is not None:
        image_time = image_time[plane::n_plane]
    else:
        # TODO used npy in combined file
        raise NotImplementedError('')

    n_image = len(image_time)
    n_frame = s2p.n_frame
    n_diff = abs(n_image - n_frame)

    if n_diff >= n_frame / 100:
        raise RuntimeError('too large difference between riglog imaging header and calcium data')

    if n_image > n_frame:
        # drop latest, mostly this case
        image_time = image_time[:n_frame]
    elif n_image < n_frame:
        # padding at the tail
        image_time = np.hstack((image_time, [image_time[-1] + (i + 1) / image_fr for i in range(n_diff)]))

    return image_time
