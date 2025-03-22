"""
OASIS
========

**Fast online deconvolution of calcium imaging dat**


Method source
--------------

- `<https://github.com/j-friedrich/OASIS>`_

- `<https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005423>`_


This script adapted from ``suite2p.suite2p.extraction.dcnv``
(Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu)

.. seealso:: `<https://github.com/MouseLand/suite2p/blob/main/suite2p/extraction/dcnv.py>`_


**Example of usage**

.. code-block:: python

    from neuralib.imaging.spikes.oasis import oasis_dcnv

    # 2D dF/F array. Array[float, [nNeurons, nFrames]] or Array[float, nFrames]
    dff = ...

    tau = 1.5  # time constant of the calcium indicator (ms)
    fs = 30  # sampling frequency of the calcium imaging data (hz)
    spks = oasis_dcnv(dff, tau, fs)


"""
import numpy as np
from numba import njit, prange

__all__ = ['oasis_dcnv',
           'oasis_matrix']


def oasis_dcnv(dff: np.ndarray,
               tau: float,
               fs: float,
               batch_size: int = 300) -> np.ndarray:
    """
    Computes non-negative deconvolution (no sparsity constraints)

    :param dff: The observed calcium fluorescence trace in 2D numpy array(multiple neurons)
        or 1D numpy array(single cell). `Array[float, [N, F]|F]`
    :param tau: The time constant of the calcium indicator in ms
    :param fs: The sampling frequency of the calcium imaging data in hz
    :param batch_size: number of frames processed per batch
    :return: Deconvolved fluorescence. `Array[float, [N,F]|F]`
    """
    dff = dff.astype(np.float32)
    if dff.ndim == 1:  # single neurons
        n_neurons = 1
        dff = np.expand_dims(dff, 0)
    else:
        n_neurons = dff.shape[0]

    n_frames = dff.shape[1]

    if batch_size > n_frames:
        batch_size = n_frames

    ret = np.zeros(dff.shape, dtype=np.float32)
    for i in range(0, n_neurons, batch_size):
        f = dff[i:i + batch_size]
        v = np.zeros(dff.shape, dtype=np.float32)
        w = np.zeros(dff.shape, dtype=np.float32)
        t = np.zeros(dff.shape, dtype=np.int64)
        ll = np.zeros(dff.shape, dtype=np.float32)
        s = np.zeros(dff.shape, dtype=np.float32)
        oasis_matrix(f, v, w, t, ll, s, tau, fs)
        ret[i:i + batch_size] = s

    if n_neurons == 1:
        return ret[0]
    else:
        return ret


@njit(parallel=True, cache=True)
def oasis_matrix(
        dff: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        t: np.ndarray,
        ll: np.ndarray,
        s: np.ndarray,
        tau: float,
        fs: float
):
    """ Performs spike deconvolution for a single neuron's calcium imaging trace using a greedy method

    Iterates through each time point in the observed fluorescence signal ``dff`` and enforces
    a non-negative and non-increasing constraint on the estimated signal.
    When a violation of this constraint is detected, the function merges the current segment with the previous one,
    updating the estimated signal accordingly. Finally, it computes the inferred spikes by
    calculating the difference between successive segments in the deconvolved signal.

    :param dff:  The observed calcium fluorescence trace. `Array[float, [N, F]|F]`
    :param v: A 1D array that will store the estimated deconvolved signal.
    :param w: A 1D array that tracks the weights for merging steps
    :param t: A 1D array that stores the indices of time steps
    :param ll: A 1D array that tracks the weights for merging steps.
    :param s: A 1D array that will store the inferred spikes
    :param tau: The time constant of the calcium indicator
    :param fs: The sampling frequency of the calcium imaging data

    """
    n_neurons = dff.shape[0]
    for n in prange(n_neurons):
        _oasis_trace(dff[n], v[n], w[n], t[n], ll[n], s[n], tau, fs)


@njit(cache=True)
def _oasis_trace(dff, v, w, t, ll, s, tau, fs):
    """single neurons spike deconvolution"""
    nframes = len(dff)
    g = -1. / (tau * fs)

    i = 0
    ip = 0

    while i < nframes:
        v[ip] = dff[i]
        w[ip] = 1
        t[ip] = i
        ll[ip] = 1

        while ip > 0:
            if v[ip - 1] * np.exp(g * ll[ip - 1]) > v[ip]:
                # violation of the constraint means merging pools
                f1 = np.exp(g * ll[ip - 1])
                f2 = np.exp(2 * g * ll[ip - 1])
                wnew = w[ip - 1] + w[ip] * f2
                v[ip - 1] = (v[ip - 1] * w[ip - 1] + v[ip] * w[ip] * f1) / wnew
                w[ip - 1] = wnew
                ll[ip - 1] = ll[ip - 1] + ll[ip]
                ip -= 1
            else:
                break
        i += 1
        ip += 1

    s[t[1:ip]] = v[1:ip] - v[:ip - 1] * np.exp(g * ll[:ip - 1])
